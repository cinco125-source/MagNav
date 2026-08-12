##* Hager et al. 2026 — to-the-letter reimplementation of the online EKF+TL+NN
##* calibrator, built so the manuscript's "we beat a tuned NN baseline" claim is
##* made against THEIR filter rather than a recipe-level lookalike.
#
# WHY THIS FILE EXISTS. research/ekf_tlnn.jl is a recipe-level reproduction and
# the audit against the preprint (research/hager_spec.md) found it differs in
# eight places, two of them architectural:
#
#   1. its NN input is 3 permanent direction cosines; theirs is 4 (the vector
#      components AND the uncompensated scalar magnetometer itself), and
#   2. it has NO Tolles-Lawson states at all -- the NN sits in MagNav.jl's TL
#      slot and REPLACES the linear model, whereas the paper's headline design
#      is additive: A_t'beta_TL carries the physics and the NN carries only the
#      residual on top of it.
#
# MagNav.jl cannot express (2). `ekf_online_nn` builds its Jacobian as
#   H1 = [Hll[1:2]; zeros(nx-3-nx_nn); Hnn.*y_scale; 1]
# which leaves no room for an A_t block, so TL states and NN weights cannot
# coexist. Hence a filter loop written out here rather than a new set of
# arguments to the library one.
#
# WHAT IS FAITHFUL. State vector, additive TL+NN measurement model, 18-term TL
# with the constant bias split out as its own state S_CB, tanh network with a
# bias-free linear output, Nh = 5, Glorot init at gain 1e-2, P0_TL = 1e5 I,
# Q_TL = I, P0_NN = I, Q_NN = 1e-20 I, fixed output denormalization alpha = 400
# nT with no offset, and chi-square innovation gating at 6 switched on only
# after the first 10 minutes.
#
# THE NOISE MODEL NEEDS NO DECISION, AND THAT IS ITSELF THE FINDING. Every
# number the authors sent back is a MagNav.jl `create_model` default:
#
#   quantity            their reply    create_model default        ours
#   position prior          3 m            3.0                     0.1
#   velocity prior          0.01 m/s       0.01                    1.0
#   FOGM tau                600 s          600.0                   180.0
#   sqrt(Qd) velocity       7.52622e-05    0.000238*sqrt(dt)       same
#   sqrt(Qd) tilt           1.83728e-07    5.81e-07*sqrt(dt)       same
#   sqrt(Qd) accel bias     1.82612e-06    2.45e-04*sqrt(2dt/3600) same
#   sqrt(Qd) gyro bias      5.41874e-11    7.27e-09*sqrt(2dt/3600) same
#
# (their Q figures are per-step standard deviations, i.e. sqrt of the diagonal of
# `create_Qd(0.1)`, with the library's baro/accel/gyro tau = 3600 s.)
#
# So the process noise was never a difference between us -- we inherited the same
# defaults. The three PRIORS were, and all three deviations are ours: we had
# tightened position by 30x, loosened velocity by 100x, and shortened the FOGM
# correlation time. Those were our tuning choices, made for our own runs, and
# they are what the P0-matched re-run moved. This file uses their values, which
# are simply the library's.
#
# WHAT IS NOT, AND WHY (each is a knob, so the cost of each is measurable):
#
#   - Vector-magnetometer pseudo-states. They carry m(3) driven by the
#     measurement through a first-order lag with tau -> 0, so in steady state
#     m equals the measurement exactly and A_t / phi_t are the same numbers we
#     compute directly from flux. The only thing their formulation buys is that
#     vector-mag noise enters the covariance. `vec_pseudo=true` adds the three
#     states to reproduce that; default off, because with tau -> 0 the state
#     estimate is not affected and the 3x3 block costs time.
#   - Input normalization method. The paper says "all inputs are normalized"
#     without saying over what. Causal: mean/sd of the first `norm_win` seconds,
#     frozen thereafter. A whole-line normalization would leak the future into a
#     cold-start filter, so this is the only defensible reading, but it IS an
#     assumption -- `norm_win` exposes it.
#   - R. hager_spec.md records the table entry as "10 nT^2", which reads as a
#     VARIANCE of 10 (sigma 3.16 nT); the reply's phrasing reads as sigma = 10
#     nT (variance 100). The ambiguity is real and unresolved, so the driver
#     below runs both and reports both. Do not quote one without the other.
#   - alt / attitude priors, which their reply did not give. create_model
#     defaults (1e-3 m, 0.01 deg) stand, flagged.
#
# Usage:
#   julia --project=. research/hager_impl.jl                    # all mags, both R readings
#   julia --project=. research/hager_impl.jl --mag 5 --seed 33
#   julia --project=. research/hager_impl.jl --flux flux_d      # match our own baselines
#
# Or as a library: `include("hager_impl.jl"); hager_drms(traj,ins,mag_uc,flux,itp_mapS)`

using MagNav
using LinearAlgebra, Statistics
using Random: MersenneTwister

##* -------- published values (research/hager_spec.md; do not "tune" these) ----
const HG_NH        = 5        # hidden neurons, single layer, tanh (their sweep's best)
const HG_GAIN      = 1e-2     # Glorot gain: sigma_L = gain*sqrt(2/(fan_in+fan_out))
const HG_ALPHA     = 400.0    # NN output denormalization [nT], no offset
const HG_P0_TL     = 1e5      # P0 for the 18 TL coefficients and for S_CB
const HG_Q_TL      = 1.0      # Q_TL = I  (create_Qd takes a sigma; sigma^2*dt per step)
const HG_P0_NN     = 1.0      # P0_NN = I
const HG_Q_NN      = 1e-20    # Q_NN = 1e-20 I -> the network freezes after the transient
const HG_CHI2      = 6.0      # innovation gate on NIS
const HG_GATE_ON   = 600.0    # gate inactive for the first 10 min [s]
const HG_POS_SIGMA = 3.0      # their P0 position prior [m]   (ours was 0.1)
const HG_VEL_SIGMA = 0.01     # their P0 velocity prior [m/s] (ours was 1.0)
const HG_TERMS     = [:permanent,:induced,:eddy]   # 18 columns, bias excluded
##* ---------------------------------------------------------------------------

##* ---------------------------- the network ----------------------------------
# Written out rather than taken from Flux, for two reasons that both bite here.
#
# `MagNav.get_Hnn` walks the Zygote gradient and appends `g[i].bias` whenever it
# is not `nothing` -- but the paper's output neuron is bias-FREE, where Flux
# stores the bias as `false`, so that path is at best fragile. And this filter
# takes a Jacobian at every one of ~52,000 epochs; an autodiff call per epoch is
# minutes of the run spent differentiating a five-neuron network.
#
# Layout of the flat weight vector w (this is the NN state block, in order):
#   w[1 : Nh*Nf]                       W1, column-major (Nh x Nf)
#   w[Nh*Nf+1 : Nh*Nf+Nh]              b1 (Nh)
#   w[Nh*Nf+Nh+1 : Nh*Nf+2Nh]          W2 (Nh), no output bias
#
#   y(phi) = W2 . tanh(W1 phi + b1)
#
#   dy/dW1[i,j] = W2[i] (1 - tanh^2(z_i)) phi_j
#   dy/db1[i]   = W2[i] (1 - tanh^2(z_i))
#   dy/dW2[i]   =        tanh(z_i)
#
# The ordering never leaves this file, so it cannot drift out of step with a
# library convention the way a `destructure` order can.

nn_np(Nf::Int, Nh::Int) = Nh*Nf + 2*Nh

nn_unpack(w, Nf::Int, Nh::Int) =
    (reshape(view(w, 1:Nh*Nf), Nh, Nf),
     view(w, Nh*Nf+1 : Nh*Nf+Nh),
     view(w, Nh*Nf+Nh+1 : Nh*Nf+2*Nh))

"""
    nn_eval(w, phi, Nf, Nh) -> (y, dy_dw)

Forward value and the gradient with respect to the weights, in one pass.
"""
function nn_eval(w::AbstractVector, phi::AbstractVector, Nf::Int, Nh::Int)
    (W1,b1,W2) = nn_unpack(w, Nf, Nh)
    z = W1*phi .+ b1
    a = tanh.(z)
    y = dot(W2, a)

    g = similar(w)
    d = W2 .* (1 .- a.^2)                       # Nh
    @inbounds for j = 1:Nf, i = 1:Nh
        g[(j-1)*Nh + i] = d[i] * phi[j]         # dW1, column-major
    end
    g[Nh*Nf+1 : Nh*Nf+Nh]        .= d           # db1
    g[Nh*Nf+Nh+1 : Nh*Nf+2*Nh]   .= a           # dW2
    return (y, g)
end

"""
    hager_w0(Nf, Nh; gain=HG_GAIN)

Glorot initialization at the paper's gain: sigma_L = gain*sqrt(2/(fan_in+fan_out)),
biases zero. Drawn uniform on +-gain*sqrt(6/(fan_in+fan_out)), which has exactly
that standard deviation -- the same convention Flux's `glorot_uniform` uses, so
this matches both the paper's formula and the library it was developed against.
"""
function hager_w0(Nf::Int, Nh::Int; gain::Real=HG_GAIN, seed::Int=33)
    rng = MersenneTwister(seed)
    w   = zeros(Float64, nn_np(Nf,Nh))
    lim1 = gain*sqrt(6/(Nf+Nh))
    lim2 = gain*sqrt(6/(Nh+1))
    w[1:Nh*Nf]                    = lim1 .* (2 .* rand(rng, Nh*Nf) .- 1)
    w[Nh*Nf+Nh+1 : Nh*Nf+2*Nh]    = lim2 .* (2 .* rand(rng, Nh)    .- 1)
    return w                                    # b1 left at zero
end

"""
    hager_ekf(traj, ins, mag_uc, flux, itp_mapS; kwargs...)

The filter. State layout (nx = 17 + 1 + 18 + Np [+ 3]):

| index            | state                                            |
|:-----------------|:-------------------------------------------------|
| 1:3              | position error (lat, lon, alt)                   |
| 4:6              | velocity error                                   |
| 7:9              | tilt                                             |
| 10:11            | baro-damped vertical channel                     |
| 12:14            | accelerometer bias                               |
| 15:17            | gyroscope bias                                   |
| 18               | S_CB, constant scalar-magnetometer bias [nT]     |
| 19:36            | beta_TL, 18 Tolles-Lawson coefficients           |
| 37:36+Np         | Lambda_NN, network weights                       |
| (36+Np+1):(+3)   | vector-mag pseudo-states, only if `vec_pseudo`   |

There is no FOGM catch-all: the paper does not have one, and with S_CB plus a
free NN there is nothing left for it to absorb. That is `fogm_state=false`
throughout, which is why the library's `get_h`/`get_H` (both hardcode `x[end]`
as the FOGM) are not used here.

Measurement:  meas_t = map(p + dp) [+ IGRF] + S_CB + A_t' beta_TL + NN(phi_t) alpha

**Returns:** named tuple `(x, P_dia, P_pos, r, nis, gated, drms, lat, lon)`.
"""
function hager_ekf(traj, ins, mag_uc, flux, itp_mapS;
                   R_var::Real      = 10.0,
                   Nh::Int          = HG_NH,
                   seed::Int        = 33,
                   norm_win::Real   = 300.0,
                   warm::Real       = 600.0,
                   core::Bool       = true,
                   vec_pseudo::Bool = false,
                   date             = get_years(2020,185))

    N  = traj.N
    dt = traj.dt

    # ---- Tolles-Lawson design matrix: 18 columns, bias column excluded ------
    A = create_TL_A(flux; terms=HG_TERMS)
    size(A,2) == 18 || error("expected 18 TL columns, got $(size(A,2))")

    # ---- NN features: the 3 vector components AND the scalar magnetometer ---
    # This is delta #2 from ekf_tlnn.jl. Feeding the map-matched scalar to the
    # network is endogenous -- it is the same signal the position update reads --
    # and in our earlier runs that collapsed. Their Q_NN = 1e-20 is what makes it
    # safe: after the transient the weights are frozen, so the endogenous channel
    # cannot keep chasing map signal. Raising HG_Q_NN reopens it; that is the
    # experiment, not a bug.
    phi = Float64.([flux.x flux.y flux.z mag_uc])
    Nf  = size(phi,2)

    # causal normalization from the first `norm_win` seconds only
    Mw   = max(2, min(N, round(Int, norm_win/dt)))
    mu   = vec(mean(phi[1:Mw,:], dims=1))
    sd   = vec(std( phi[1:Mw,:], dims=1))
    sd[sd .< 1e-12] .= 1.0
    phi  = (phi .- mu') ./ sd'

    # ---- network ------------------------------------------------------------
    w0 = hager_w0(Nf, Nh; seed=seed)
    Np = length(w0)

    # ---- state layout -------------------------------------------------------
    nx_blk = 1 + 18 + Np                 # S_CB, beta_TL, Lambda_NN
    iCB    = 18
    iTL    = 19:36
    iNN    = 37:(36+Np)

    P0_blk = zeros(Float64, nx_blk, nx_blk)
    P0_blk[1,1]         = HG_P0_TL                      # S_CB
    P0_blk[2:19,2:19]   = HG_P0_TL * I(18)              # beta_TL
    P0_blk[20:end,20:end] = HG_P0_NN * I(Np)            # Lambda_NN
    sig_blk = [0.0; fill(HG_Q_TL,18); fill(sqrt(HG_Q_NN),Np)]  # S_CB constant

    (P0,Qd,R) = create_model(dt, traj.lat[1];
                             init_pos_sigma = HG_POS_SIGMA,
                             init_vel_sigma = HG_VEL_SIGMA,
                             meas_var       = R_var,
                             TL_sigma       = sig_blk,
                             P0_TL          = P0_blk,
                             vec_states     = vec_pseudo,
                             fogm_state     = false)

    nx = size(P0,1)
    nx == 17 + nx_blk + (vec_pseudo ? 3 : 0) || error("state layout mismatch: nx = $nx")

    x = zeros(Float64, nx)
    x[iNN] .= w0                      # beta_TL,0 = 0 and S_CB,0 = 0; only the net is seeded
    P = copy(P0)

    x_out = zeros(Float64, nx, N)
    # diagonal only: a full P history would be nx^2*N*8 = 1.8 GB at nx = 66 over
    # an 87-minute line, and nothing downstream reads the off-diagonals
    P_dia = zeros(Float64, nx, N)
    P_pos = zeros(Float64, 2, 2, N)     # the position block, kept for NEES
    r_out = zeros(Float64, N)
    nis   = zeros(Float64, N)
    gated = falses(N)

    Hbase = zeros(Float64, nx)
    Hbase[iCB] = 1.0

    for t = 1:N
        Phi = MagNav.get_Phi(nx, ins.lat[t], ins.vn[t], ins.ve[t], ins.vd[t],
                             ins.fn[t], ins.fe[t], ins.fd[t], ins.Cnb[:,:,t],
                             3600.0, 3600.0, 3600.0, 600.0, dt;
                             vec_states = vec_pseudo,
                             fogm_state = false)

        lat_ = ins.lat[t] + x[1]
        lon_ = ins.lon[t] + x[2]
        alt_ = ins.alt[t] + x[3]

        (y_nn, g_nn) = nn_eval(view(x,iNN), view(phi,t,:), Nf, Nh)

        h = itp_mapS(lat_, lon_, alt_) + x[iCB] + dot(A[t,:], x[iTL]) + y_nn*HG_ALPHA
        core && (h += norm(MagNav.igrf(date, alt_, lat_, lon_, Val(:geodetic))))
        resid = mag_uc[t] - h

        H = copy(Hbase)
        H[1:3] = MagNav.map_grad(itp_mapS, lat_, lon_, alt_)
        core && (H[1:3] += MagNav.igrf_grad(lat_, lon_, alt_; date=date))
        H[iTL] = A[t,:]
        H[iNN] = g_nn .* HG_ALPHA

        S      = dot(H, P, H) + R
        nis[t] = resid^2 / S

        # chi-square gate, inactive during the transient (their choice, and it
        # matters: gating a cold start rejects the very residuals that teach the
        # network what the interference is)
        if (t-1)*dt >= HG_GATE_ON && nis[t] > HG_CHI2
            gated[t] = true
        else
            K = (P*H) / S
            x = x + K*resid
            P = (I - K*H') * P
            P = (P + P') / 2
        end

        x_out[:,t]     = x
        P_dia[:,t]     = diag(P)
        P_pos[:,:,t]   = P[1:2,1:2]
        r_out[t]       = resid

        x = Phi*x
        P = Phi*P*Phi' + Qd
    end

    lat_est = ins.lat .+ x_out[1,:]
    lon_est = ins.lon .+ x_out[2,:]
    m0  = min(round(Int, warm/dt) + 1, N)
    ind = m0:N
    drms = sqrt(mean(dlat2dn.(lat_est[ind].-traj.lat[ind], traj.lat[ind]).^2 .+
                     dlon2de.(lon_est[ind].-traj.lon[ind], traj.lat[ind]).^2))

    return (x=x_out, P_dia=P_dia, P_pos=P_pos, r=r_out, nis=nis, gated=gated,
            drms=drms, lat=lat_est, lon=lon_est)
end

"""
    hager_drms(traj, ins, mag_uc, flux, itp_mapS; kwargs...)

Horizontal DRMS [m] after `warm` seconds, `Inf` on divergence or error -- the
same contract as `ekf_tlnn_drms` in research/ekf_tlnn.jl, so the two can be
dropped into the same breadth harness and compared column against column.
"""
function hager_drms(traj, ins, mag_uc, flux, itp_mapS;
                    warm::Real=600.0, div_thresh::Real=1e4, kwargs...)
    try
        out = hager_ekf(traj, ins, mag_uc, flux, itp_mapS; warm=warm, kwargs...)
        d   = out.drms
        return (isfinite(d) && d <= div_thresh) ? d : Inf
    catch e
        @warn("hager_drms failed", e)
        return Inf
    end
end

##* ---------------------------- driver ---------------------------------------
# Reproduces the paper's Table on line 1007.06 (cold start, Nh = 5):
#   Mag 2: 51   Mag 3: 42   Mag 4: 37   Mag 5: 14   Mag 1 floor: 17
# Runs both readings of R because the preprint's table entry is ambiguous.

if abspath(PROGRAM_FILE) == @__FILE__

    using CSV, DataFrames

    arg(flag, default, T=String) = begin
        i = findfirst(==(flag), ARGS)
        i === nothing ? default : parse_arg(T, ARGS[i+1])
    end
    parse_arg(::Type{String}, s) = s
    parse_arg(T, s) = parse(T, s)

    seed  = arg("--seed", 33, Int)
    mags  = "--mag" in ARGS ? [arg("--mag", 5, Int)] : [1,2,3,4,5]
    Rvars = "--R" in ARGS ? [arg("--R", 10.0, Float64)] : [10.0, 100.0]
    line  = arg("--line", 1007.06, Float64)
    # their spec says Flux A; our own baselines (paper_impl.jl, fgo_breadth.jl)
    # all ran Flux D, so this has to be switchable or the comparison is confounded
    fluxsym = Symbol(arg("--flux", "flux_a"))

    df_dir    = joinpath(@__DIR__,"..","examples","dataframes")
    df_flight = DataFrame(CSV.File(joinpath(df_dir,"df_flight.csv")))
    df_flight[!,:flight]   = Symbol.(df_flight[!,:flight])
    df_flight[!,:xyz_type] = Symbol.(df_flight[!,:xyz_type])
    df_flight[!,:xyz_file] = String.(df_flight[!,:xyz_file])
    for (i,flight) in enumerate(df_flight.flight)
        flight in (:Flt1007,) || continue
        df_flight.xyz_file[i] = MagNav.sgl_2020_train(flight)
    end
    df_map = DataFrame(CSV.File(joinpath(df_dir,"df_map.csv")))
    df_map[!,:map_name] = Symbol.(df_map[!,:map_name])
    df_map[!,:map_file] = String.(df_map[!,:map_file])
    for (i,map_name) in enumerate(df_map.map_name)
        df_map.map_file[i] = MagNav.ottawa_area_maps(map_name)
    end
    df_nav = DataFrame(CSV.File(joinpath(df_dir,"df_nav.csv")))
    df_nav[!,:flight]   = Symbol.(df_nav[!,:flight])
    df_nav[!,:map_name] = Symbol.(df_nav[!,:map_name])

    xyz  = get_XYZ(:Flt1007, df_flight; silent=true)
    ind  = get_ind(xyz, line, df_nav)
    traj = get_traj(xyz, ind)
    ins  = get_ins(xyz, ind; N_zero_ll=1)   # cold start from truth, as in paper_impl.jl
    flux = getfield(xyz, fluxsym)(ind)
    mapS = get_map(get_map_name(xyz, line, df_nav), df_map)
    (_,itp_mapS) = get_map_val(mapS, traj; return_itp=true)

    rows = DataFrame(mag=Int[], R_var=Float64[], drms=Float64[],
                     gated_frac=Float64[], nis_med=Float64[])

    println("Hager et al. 2026, faithful build -- line $line, cold start, Nh = $HG_NH, $fluxsym")
    println("published: mag2 51, mag3 42, mag4 37, mag5 14, mag1 floor 17 [m]\n")
    println(rpad("mag",5), rpad("R [nT^2]",11), rpad("DRMS [m]",11),
            rpad("gated",9), "median NIS")

    for mg in mags, Rv in Rvars
        mag_uc = getfield(xyz, Symbol("mag_$(mg)_uc"))[ind]
        out    = try
            hager_ekf(traj, ins, mag_uc, flux, itp_mapS; R_var=Rv, seed=seed)
        catch e
            @warn("mag $mg, R $Rv failed", e); nothing
        end
        if out === nothing
            println(rpad(mg,5), rpad(Rv,11), rpad("err",11))
            push!(rows, (mg, Rv, Inf, NaN, NaN))
        else
            gf = mean(out.gated)
            nm = median(out.nis)
            println(rpad(mg,5), rpad(Rv,11), rpad(round(out.drms,digits=1),11),
                    rpad(string(round(100*gf,digits=1),"%"),9), round(nm,digits=2))
            push!(rows, (mg, Rv, out.drms, gf, nm))
        end
    end

    CSV.write(joinpath(@__DIR__,"hager_impl_results.csv"), rows)
    println("\nwrote research/hager_impl_results.csv")
    println("read the gated column before the DRMS column: a gate rejecting more")
    println("than a few percent means R is the wrong reading, not that the filter won.")
end
