##* Reproduction of the online EKF + Tolles-Lawson + NN cold-start calibrator
##* of Hager et al. (2026, arXiv 2603.08265), as a REAL comparison baseline.
#
# Unlike research/paper_baseline.jl (which only CITES the paper's published
# numbers), this script actually RUNS the NN-augmented online EKF on the paper's
# primary line 1007.06 from a cold start, and tunes it toward the paper's
# reported DRMS (Mag 4: 58 TL-only / 37 TL+NN; Mag 5: 15 / 14).
#
# The MagNav.jl `ekf_online_nn` IS the reference implementation of this filter
# family (the SGL/AFIT online NN-in-EKF lineage the paper builds on):
#   resid_t = meas_t − [ NN(x_nn_t)·y_scale + y_bias ] − h_map(x_t)
# with the NN WEIGHTS carried as EKF states and learned online. Tolles-Lawson is
# embedded by including the TL A-matrix columns among the NN input features, so
# the network represents "TL + nonlinear residual" — i.e. EKF + TL + NN.
#
# Cold start = no calibration flight, no NN pre-training: the network is randomly
# initialized and the compensation is learned online from the map-match residual.
# The NN is run bias-FREE so it represents only the aircraft interference (the
# core + map field is supplied by get_h); the weight covariances (initial P0_nn,
# per-step process noise) are set by hand so the adaptation rate is an explicit,
# tunable knob rather than an opaque warm-start artifact.
#
# Data: SGL 2020, line 1007.06, full length, uncompensated cabin magnetometers,
# DRMS after a 10-min warm-up (paper convention).
#
# Usage: julia --project=. research/paper_impl.jl

using MagNav
using CSV, DataFrames
using LinearAlgebra, Statistics
using Random: seed!
seed!(33)

##* -------- tuning knobs (iterate these toward the paper band) --------------
NN_HIDDEN   = [8]                       # NN hidden layer sizes (small ⇒ memory-safe)
TL_TERMS    = [:permanent]              # TL basis columns fed to the NN as features
P0NN_SIGMA  = 0.3                       # initial NN-weight std (weights are O(1))
WEIGHT_Q    = 3e-3                      # per-step NN-weight random-walk std (adaptation rate)
MEAS_VAR    = 12.0^2                    # scalar map-match measurement variance [nT^2]
# (12^2 is the locked value: it gave the best noisy-mag result, Mag 4 40.0 m;
#  tightening to 7^2 sharpened Mag 5 slightly but regressed Mag 4 to 46.5 m.)
FOGM_SIGMA  = 3.0                       # FOGM catch-all sigma [nT]
FOGM_TAU    = 180.0                     # FOGM catch-all time constant [s]
WARMUP_S    = 600.0                     # DRMS warm-up (paper convention) [s]
##* --------------------------------------------------------------------------
# NOTE on the measurement model (why divergence is avoided): the EKF residual is
#   resid = meas − [ NN(x_nn)·y_scale + y_bias ] − get_h(map; core=true)
# get_h already supplies the core (IGRF) + map anomaly (~50000 nT), so the NN
# compensation must represent ONLY the aircraft interference (~10²–10³ nT). The
# NN bias y_bias is set to the interference DC estimated onboard over the first
# few minutes (see the per-mag block): a randomly-initialized network then starts
# near the right offset, so the first residual is small even for the noisy cabin
# mags and position is not kicked before the NN learns (true cold start). y_scale
# sets the NN output range (≈ interference std). We size the weight covariance by
# hand rather than via ekf_online_nn_setup, whose RLS warm start mixes units.

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

flight = :Flt1007
line   = 1007.06
@info("loading $flight")
xyz  = get_XYZ(flight,df_flight;silent=true)
ind  = get_ind(xyz,line,df_nav)
map_name = df_nav[(df_nav.flight.==flight).&(df_nav.line.==line),:map_name][1]
mapS = get_map(map_name,df_map)

traj = get_traj(xyz,ind)
ins  = get_ins( xyz,ind;N_zero_ll=1)
(map_val,itp_mapS) = get_map_val(mapS,traj;return_itp=true)
flux = xyz.flux_d(ind)
N    = traj.N
println("line $line: N=$N, ~$(round(N*traj.dt/60,digits=1)) min, map=$map_name")

# horizontal position DRMS [m] after a warm-up window (paper convention)
function drms(fo; warm=WARMUP_S)
    m  = (traj.tt .- traj.tt[1]) .>= warm
    dn = dlat2dn.(fo.lat[m] .- traj.lat[m], traj.lat[m])
    de = dlon2de.(fo.lon[m] .- traj.lon[m], traj.lat[m])
    sqrt(mean(dn.^2 .+ de.^2))
end
ins_drms = let m=(traj.tt.-traj.tt[1]).>=WARMUP_S
    sqrt(mean(dlat2dn.(ins.lat[m].-traj.lat[m],traj.lat[m]).^2 .+
              dlon2de.(ins.lon[m].-traj.lon[m],traj.lat[m]).^2)) end
println("\nINS (no aiding) DRMS(>10min) = ",round(ins_drms,digits=1)," m")

results = DataFrame(method=String[],mag=String[],drms=Float64[])
log!(name,mag,d) = (push!(results,(name,mag,d));
    println(rpad(name,34)," ",mag," DRMS(>10min) = ",round(d,digits=1)," m"); d)

# stinger reference: EKF on the pre-compensated Mag 1
(P0n,Qdn,Rn) = create_model(traj.dt,traj.lat[1];
                            init_pos_sigma=0.1,init_alt_sigma=1.0,init_vel_sigma=1.0,
                            meas_var=MEAS_VAR,fogm_sigma=FOGM_SIGMA,fogm_tau=FOGM_TAU)
try
    fo = run_filt(traj,ins,xyz.mag_1_c[ind],itp_mapS,:ekf;P0=P0n,Qd=Qdn,R=Rn,
                  core=true,run_crlb=false); log!("EKF (Mag1 compensated)","Mag 1",drms(fo))
catch e; @warn("EKF Mag1 failed",e) end
println()

##* cold-start online EKF+TL+NN on each uncompensated cabin magnetometer
for magsym in (:mag_4_uc, :mag_5_uc)
    mag_uc = getfield(xyz,magsym)[ind]
    tag    = replace(String(magsym),"mag_"=>"Mag ","_uc"=>"")
    println("=== $tag (uncompensated cabin magnetometer, cold start) ===")
    try
        # features: TL A-matrix columns ONLY (fluxgate/attitude-driven, map-
        # independent). The scalar mag_uc is deliberately EXCLUDED: it carries the
        # map anomaly, so feeding it to the compensation NN lets the network
        # subtract the very signal we navigate on — driving the residual to ~0
        # while position drifts (an observability collapse, seen on Mag 4 at
        # 5.8 km with max|resid| only 78 nT). With attitude-only features the NN
        # can model heading-dependent interference but cannot represent the map.
        x  = create_TL_A(flux;terms=TL_TERMS)
        Nf = size(x,2)
        (_,_,x_norm) = norm_sets(x)
        x_norm = Float32.(x_norm)            # match NN parameter eltype (avoid per-step convert)
        # Cold-start DC initialization (fixes the large-interference divergence):
        # a randomly-initialized NN outputs ~0, so with a bias-free compensation
        # the first residual is the full interference DC — for the noisy cabin
        # mags (~250 nT) that spikes the Kalman gain and runs position away before
        # the NN can learn. We remove that DC at t=0 using ONLY onboard information
        # (map anomaly + IGRF core, evaluated at the INS position over the first
        # 5 min while the INS still ≈ truth — no truth leak); the NN then only has
        # to learn the maneuver-dependent residual around it.
        Mw   = min(N, round(Int, 300/traj.dt))
        xz   = zeros(18, Mw)
        pred = MagNav.get_h(itp_mapS, xz, ins.lat[1:Mw], ins.lon[1:Mw], ins.alt[1:Mw]; core=true)
        intf = mag_uc[1:Mw] .- pred          # aircraft interference estimate [nT]
        bias0   = median(intf)               # interference DC to remove at cold start
        y_scale = std(intf)                  # NN output range ≈ interference std
        y_norms = (Float32(bias0), Float32(y_scale))
        println("  Nf=$Nf feat, interference DC=",round(bias0,digits=1),
                " nT, y_scale=",round(y_scale,digits=1)," nT")

        m  = MagNav.get_nn_m(Nf,1;hidden=NN_HIDDEN)   # randomly-initialized NN (cold start)
        nx_nn    = length(MagNav.destructure(m)[1])
        P0_nn    = Matrix(Diagonal(fill(P0NN_SIGMA^2, nx_nn)))
        nn_sigma = fill(WEIGHT_Q, nx_nn)
        println("  nx_nn=$nx_nn NN-weight states")

        (P0,Qd,R) = create_model(traj.dt,traj.lat[1];
                                 init_pos_sigma=0.1,init_alt_sigma=1.0,init_vel_sigma=1.0,
                                 meas_var=MEAS_VAR,fogm_sigma=FOGM_SIGMA,fogm_tau=FOGM_TAU,
                                 vec_states=false,TL_sigma=nn_sigma,P0_TL=P0_nn)

        frw = ekf_online_nn(ins,mag_uc,itp_mapS,x_norm,m,y_norms,P0,Qd,R;
                            fogm_tau=FOGM_TAU,core=true)
        fo  = MagNav.eval_filt(traj,ins,frw)
        # diagnostic: DRMS over the last half of the flight — if the NN has
        # learned the compensation, the late-window error is much smaller than
        # the full-warmup error even when the cold-start transient was rough.
        late = round(drms(fo;warm=N*traj.dt/2),digits=1)
        rmax = round(maximum(abs,frw.r),digits=0)
        println("  late-half DRMS = $late m,  max|resid| = $rmax nT")
        log!("EKF+TL+NN online (cold start)",tag,drms(fo))
    catch e; @warn("ekf_online_nn failed for $tag",e) end
    println()
end

##* paper's PUBLISHED cold-start DRMS on line 1007.06 (Hager et al. 2026)
paper = DataFrame(mag=["Mag 1","Mag 3","Mag 4","Mag 5"],
                  paper_TL_only=[17.0,46.0,58.0,15.0],
                  paper_TL_NN  =[17.0,42.0,37.0,14.0])

println("=== OUR reproduced EKF+TL+NN (DRMS >10min, line $line, full length) ===")
show(results;allrows=true,allcols=true); println()
println("\n=== paper reported cold-start DRMS [m] (Hager et al. 2026) ===")
show(paper;allrows=true,allcols=true); println()
CSV.write(joinpath(@__DIR__,"paper_impl_results.csv"),results)
println("\nknobs: hidden=$NN_HIDDEN terms=$TL_TERMS P0nn=$P0NN_SIGMA weight_q=$WEIGHT_Q ",
        "meas_var=$MEAS_VAR fogm=($FOGM_SIGMA,$FOGM_TAU)")
