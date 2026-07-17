##* MPF + TL + NN: the particle filter given the same neural network as the EKF.
#
# The breadth study showed the NN is what keeps the causal EKF bounded at cold
# start, while the NN-less MPF+TL diverges. The natural completion of the 2x2
# (host filter x compensation) is to give the marginalized particle filter the
# SAME online NN as ekf_online_nn / ekf_tlnn: TL-permanent features, cold-start
# DC init, weights carried in the conditionally-Gaussian block.
#
# One approximation is forced and worth stating: the NN output is nonlinear in
# its weights, so the weights are NOT conditionally linear-Gaussian; as in
# ekf_online_nn we linearize about the current weight estimate (here the
# particle-weighted mean), which makes the marginalized block an EKF block.
# Per particle the predicted interference uses the same first-order expansion
# NN(w_p) ~ NN(w_bar) + Hnn'(w_p - w_bar), consistent with that linearization.
#
# Runs the 4 counted breadth lines x Mag 4/5; outputs research/mpf_nn_results.csv.
# Usage: julia --project=. research/mpf_nn.jl

using MagNav
using CSV, DataFrames
using LinearAlgebra, Statistics
using Random: randn, seed!
using Flux
seed!(33)

# locked hyperparameters, identical to research/ekf_tlnn.jl (the EKF+TL+NN recipe)
const NN_HIDDEN   = [8]
const NN_TERMS    = [:permanent]
const NN_P0_SIGMA = 0.3
const NN_WEIGHT_Q = 3e-3
const MEAS_VAR    = 12.0^2
const FOGM_SIG    = 3.0
const FOGM_TAU    = 180.0
const NUM_PART    = 300     # matches the MPF+TL baseline in fgo_breadth.jl

function mpf_online_nn(ins::MagNav.INS, meas, x_nn, m0, y_norms, P0, Qd, R;
                       num_part = NUM_PART,
                       thresh   = 0.8,
                       baro_tau = 3600.0, acc_tau = 3600.0, gyro_tau = 3600.0,
                       fogm_tau = FOGM_TAU,
                       warm_infl = 300.0,   # [s] cold-start R inflation window
                       R_gain    = 1.0e3,
                       date     = MagNav.get_years(2020,185),
                       core::Bool = true,
                       itp_mapS = nothing)

    lat=ins.lat; lon=ins.lon; alt=ins.alt; vn=ins.vn; ve=ins.ve; vd=ins.vd
    fn=ins.fn; fe=ins.fe; fd=ins.fd; Cnb=ins.Cnb; dt=ins.dt
    N   = length(lat); np = num_part
    nx  = size(P0,1)
    (w0,re) = Flux.destructure(m0)
    nx_nn = length(w0)
    nxn = 2; nxl = nx - nxn                     # (lat,lon) nonlinear; rest linear
    ny  = size(meas,2); T2 = eltype(P0)
    (y_bias,y_scale) = T2.(y_norms)
    iw  = (18-nxn):(17+nx_nn-nxn)               # weight rows within linear block
    npl = nxl - nx_nn - 1                        # Pinson linear rows before weights

    x_out  = zeros(T2,nx,N)
    Pn_out = zeros(T2,nxn,nxn,N)
    Pl_out = zeros(T2,nxl,nxl,N)
    resid  = zeros(T2,ny,N)

    xn = MagNav.chol(P0[1:nxn,1:nxn])'*randn(T2,nxn,np)
    xl = zeros(T2,nxl,np); xl[iw,:] .= w0        # all particles start at w0
    Pl = P0[nxn+1:end,nxn+1:end]
    q  = ones(T2,np)/np

    for t = 1:N
        Phi  = MagNav.get_Phi(nx,lat[t],vn[t],ve[t],vd[t],fn[t],fe[t],fd[t],
                       Cnb[:,:,t],baro_tau,acc_tau,gyro_tau,fogm_tau,dt)
        An_l = Phi[1:nxn,nxn+1:end]; An_n = Phi[1:nxn,1:nxn]
        Al_l = Phi[nxn+1:end,nxn+1:end]; Al_n = Phi[nxn+1:end,1:nxn]

        # linearize the NN about the particle-weighted mean weights
        wbar = vec(sum(reshape(q,1,np).*xl[iw,:],dims=2))
        mnet = re(wbar)
        xt   = x_nn[t,:]
        nn0  = T2(mnet(xt)[1])
        Hnn  = T2.(MagNav.get_Hnn(MagNav.nn_grad(mnet,xt)))
        H    = [zeros(T2,1,npl) reshape(Hnn.*y_scale,1,nx_nn) ones(T2,1,1)]  # 1 x nxl

        # per-particle prediction: map(+core+S+pos via get_h) + linearized NN
        hmap = MagNav.get_h(itp_mapS,[xn;xl],lat[t],lon[t],alt[t];date=date,core=core)
        cnn  = (nn0 .+ vec(Hnn'*(xl[iw,:] .- wbar))).*y_scale .+ y_bias
        yhat = hmap .+ cnn

        e = repeat(meas[t,:],1,np) - repeat(yhat',ny,1)
        resid[:,t] = mean(e,dims=2)

        # cold-start defense (as in mpf_online, hardened): geometric decay of the
        # R inflation (no cliff) and log-domain weights with a max-shift so a
        # large common residual cannot underflow every particle at once.
        gain = max(one(T2), T2(R_gain)^(1 - (t-1)*dt/(3*warm_infl)))
        Rt = R .* gain
        V  = H*Pl*H' .+ Rt
        logw = zeros(T2,np)
        for i = 1:ny
            logw .+= -0.5.*(e[i,:].*(1/V[i,i]).*e[i,:])
        end
        q = q .* exp.(logw .- maximum(logw))
        if sum(q) > eps(T2)
            q = q/sum(q)
            for i = 1:nxn; x_out[i,t] = sum(q.*xn[i,:]); end
            Pn_out[:,:,t] = MagNav.part_cov(q,xn,x_out[1:nxn,t])
            if 1/sum(q.^2) < np*thresh
                ind = MagNav.sys_resample(q); xn = xn[:,ind]; xl = xl[:,ind]
                q = ones(T2,np)/np
            end
        else
            return MagNav.FILTres(x_out, MagNav.filter_exit(Pl_out,Pn_out,t,false), resid, false)
        end

        K  = Pl * H' / V
        Pl = Pl - K*V*K'

        e  = repeat(meas[t,:],1,np) - repeat(yhat',ny,1)
        xl_temp = xl
        xl = xl + K*e
        for i = 1:nxl; x_out[nxn+i,t] = sum(q.*xl[i,:]); end
        Pl_out[:,:,t] = MagNav.part_cov(q,xl,x_out[1+nxn:end,t],Pl)

        M  = An_l*Pl*An_l' + Qd[1:nxn,1:nxn]
        L  = Al_l*Pl*An_l' / M
        Pl = Al_l*Pl*Al_l' + Qd[nxn+1:end,nxn+1:end] - L*M*L'
        Pl = (Pl+Pl')/2

        xn_temp = xn
        xn = An_n*xn_temp + An_l*xl_temp + MagNav.chol(M)'*randn(T2,nxn,np)
        z  = xn - An_n*xn_temp
        xl = Al_n*xn_temp + Al_l*xl + L*(z-An_l*xl)
    end

    return MagNav.FILTres(x_out, MagNav.filter_exit(Pl_out,Pn_out,N,true), resid, true)
end

# DRMS wrapper with the exact ekf_tlnn.jl cold-start recipe (features, DC init).
function mpf_nn_drms(traj, ins, mag_uc, flux, itp; warm=600.0, div_thresh=1e4)
    try
        N  = traj.N
        x  = create_TL_A(flux; terms=NN_TERMS)
        Nf = size(x,2)
        (_,_,x_norm) = norm_sets(x)
        x_norm = Float32.(x_norm)

        Mw   = min(N, round(Int, 300/traj.dt))
        xz   = zeros(18, Mw)
        pred = MagNav.get_h(itp, xz, ins.lat[1:Mw], ins.lon[1:Mw], ins.alt[1:Mw]; core=true)
        intf = mag_uc[1:Mw] .- pred
        y_norms = (Float32(median(intf)), Float32(std(intf)))

        m0    = MagNav.get_nn_m(Nf,1; hidden=NN_HIDDEN)
        nx_nn = length(Flux.destructure(m0)[1])
        P0_nn = Matrix(Diagonal(fill(NN_P0_SIGMA^2, nx_nn)))
        nnsig = fill(NN_WEIGHT_Q, nx_nn)

        (P0,Qd,R) = create_model(traj.dt,traj.lat[1];
                                 init_pos_sigma=0.1,init_alt_sigma=1.0,init_vel_sigma=1.0,
                                 meas_var=MEAS_VAR,fogm_sigma=FOGM_SIG,
                                 fogm_tau=FOGM_TAU,vec_states=false,
                                 TL_sigma=nnsig,P0_TL=P0_nn)

        fr = mpf_online_nn(ins,mag_uc,x_norm,m0,y_norms,P0,Qd,R;itp_mapS=itp)
        fr.c || return Inf
        fo = MagNav.eval_filt(traj,ins,fr)
        mk = (traj.tt .- traj.tt[1]) .>= warm
        dn = dlat2dn.(fo.lat[mk] .- traj.lat[mk], traj.lat[mk])
        de = dlon2de.(fo.lon[mk] .- traj.lon[mk], traj.lat[mk])
        d  = sqrt(mean(dn.^2 .+ de.^2))
        return (isfinite(d) && d < div_thresh) ? d : Inf
    catch e
        @warn("mpf_nn_drms failed",e); return Inf
    end
end

##* run the 4 counted breadth lines ---------------------------------------------
LINES = [(:Flt1003,1003.02), (:Flt1003,1003.08),
         (:Flt1007,1007.02), (:Flt1007,1007.06)]
flights = unique(first.(LINES))

df_dir    = joinpath(@__DIR__,"..","examples","dataframes")
df_flight = DataFrame(CSV.File(joinpath(df_dir,"df_flight.csv")))
df_flight[!,:flight]   = Symbol.(df_flight[!,:flight])
df_flight[!,:xyz_type] = Symbol.(df_flight[!,:xyz_type])
df_flight[!,:xyz_file] = String.(df_flight[!,:xyz_file])
for (i,fl) in enumerate(df_flight.flight)
    fl in flights || continue
    df_flight.xyz_file[i] = MagNav.sgl_2020_train(fl)
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

xyz_cache = Dict{Symbol,Any}()
getxyz(fl) = get!(()->get_XYZ(fl,df_flight;silent=true), xyz_cache, fl)

results = DataFrame(flight=Symbol[],line=Float64[],mag=String[],
                    MPF_TLNN=Float64[],N=Int[],t=Float64[])
for (fl,line) in LINES
    xyz   = getxyz(fl)
    ind   = get_ind(xyz,line,df_nav)
    mname = df_nav[(df_nav.flight.==fl).&(df_nav.line.==line),:map_name][1]
    mapS  = get_map(mname,df_map)
    traj  = get_traj(xyz,ind)
    ins   = get_ins(xyz,ind;N_zero_ll=1)
    (_,itp) = get_map_val(mapS,traj;return_itp=true)
    flux  = xyz.flux_d(ind)
    println("\n$fl $line ($mname, ~$(round(traj.N*traj.dt/60,digits=0)) min)")
    for magsym in (:mag_4_uc,:mag_5_uc)
        mag = getfield(xyz,magsym)[ind]
        tag = replace(String(magsym),"mag_"=>"Mag ","_uc"=>"")
        tel = @elapsed (d = mpf_nn_drms(traj,ins,mag,flux,itp))
        push!(results,(fl,line,tag,round(d,digits=1),traj.N,round(tel,digits=2)))
        println("  $tag  MPF+TL+NN DRMS=$(round(d,digits=1)) m  ($(round(tel,digits=1)) s)")
    end
end

println("\n=== MPF+TL+NN (particle filter given the EKF's neural network) ===")
show(results;allrows=true,allcols=true); println()
CSV.write(joinpath(@__DIR__,"mpf_nn_results.csv"),results)
println("wrote mpf_nn_results.csv")
