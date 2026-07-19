##* MPF + online Tolles-Lawson: a fair particle-filter cold-start baseline.
#
# MagNav.jl's marginalized (Rao-Blackwellized) particle filter (`mpf`) map-matches
# but has no online compensation, so on the uncompensated cabin magnetometers it
# has nothing to remove the aircraft field and diverges like the plain EKF. Because
# the Tolles-Lawson interference is *linear* in the coefficients beta, beta is a
# conditionally-linear-Gaussian state and fits the RBPF exactly: we carry it in the
# marginalized (Kalman) block and put the fluxgate row A_t into the linear
# measurement Jacobian. This mirrors the augmented state of ekf_online
# ([Pinson-17; beta; S], S last) and MagNav's mpf recursion.
#
# mpf_online(ins, meas, flux, itp_mapS, x0_TL, P0, Qd, R; terms, num_part, ...)
# returns a FILTres, so run it through eval_filt for DRMS like the other filters.

using MagNav
using LinearAlgebra
using Random: randn
using Statistics: mean

function mpf_online(ins::MagNav.INS, meas, flux::MagNav.MagV, itp_mapS, x0_TL, P0, Qd, R;
                    terms    = [:permanent,:induced,:eddy,:bias],
                    num_part = 1000,
                    thresh   = 0.8,
                    roughen_m = 0.0,     # [m] position jitter reinjected after resample
                    baro_tau = 3600.0,
                    acc_tau  = 3600.0,
                    gyro_tau = 3600.0,
                    fogm_tau = 600.0,
                    warm_infl = 300.0,   # [s] cold-start window with inflated R
                    R_gain    = 1.0e3,   # measurement-variance inflation factor
                    date     = MagNav.get_years(2020,185),
                    core::Bool = false)

    lat=ins.lat; lon=ins.lon; alt=ins.alt; vn=ins.vn; ve=ins.ve; vd=ins.vd
    fn=ins.fn; fe=ins.fe; fd=ins.fd; Cnb=ins.Cnb; dt=ins.dt
    N   = length(lat); np = num_part
    nx  = size(P0,1); nx_TL = length(x0_TL)
    nxn = 2; nxl = nx - nxn         # position (lat,lon) nonlinear; rest linear
    ny  = size(meas,2); T2 = eltype(P0)

    A   = create_TL_A(flux; terms=terms)          # N x nx_TL, aircraft-field basis
    ib  = (18-nxn):(17+nx_TL-nxn)                 # beta indices within the linear block
    npl = nxl - nx_TL - 1                          # Pinson linear states before beta

    x_out  = zeros(T2,nx,N)
    Pn_out = zeros(T2,nxn,nxn,N)
    Pl_out = zeros(T2,nxl,nxl,N)
    resid  = zeros(T2,ny,N)

    xn = MagNav.chol(P0[1:nxn,1:nxn])'*randn(T2,nxn,np)
    xl = zeros(T2,nxl,np); xl[ib,:] .= x0_TL      # cold-start beta = x0_TL
    Pl = P0[nxn+1:end,nxn+1:end]
    q  = ones(T2,np)/np

    yhat(t) = MagNav.get_h(itp_mapS,[xn;xl],lat[t],lon[t],alt[t];date=date,core=core) .+
              vec(A[t,:]' * xl[ib,:])              # map + S (get_h) + A_t' beta

    for t = 1:N
        Phi  = MagNav.get_Phi(nx,lat[t],vn[t],ve[t],vd[t],fn[t],fe[t],fd[t],Cnb[:,:,t],
                       baro_tau,acc_tau,gyro_tau,fogm_tau,dt)
        An_l = Phi[1:nxn,nxn+1:end]; An_n = Phi[1:nxn,1:nxn]
        Al_l = Phi[nxn+1:end,nxn+1:end]; Al_n = Phi[nxn+1:end,1:nxn]

        H = [zeros(T2,1,npl) reshape(A[t,:],1,nx_TL) ones(T2,1,1)]   # 1 x nxl

        e = repeat(meas[t,:],1,np) - repeat(yhat(t)',ny,1)
        resid[:,t] = mean(e,dims=2)

        # cold-start defense: with beta=0 the first residuals are hundreds of nT.
        # Two safeguards (both standard PF practice, added after review): the R
        # inflation DECAYS geometrically instead of ending in a cliff, and the
        # weight update is done in the log domain with a max-shift so a large
        # common residual cannot underflow every particle simultaneously (only
        # RELATIVE likelihoods matter).
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
                if roughen_m > 0   # regularized-PF roughening: reinject position diversity
                    xn[1,:] .+= MagNav.dn2dlat(roughen_m, lat[t]).*randn(T2,np)
                    xn[2,:] .+= MagNav.de2dlon(roughen_m, lat[t]).*randn(T2,np)
                end
            end
        else
            return MagNav.FILTres(x_out, MagNav.filter_exit(Pl_out,Pn_out,t,false), resid, false)
        end

        K  = Pl * H' / V
        Pl = Pl - K*V*K'

        e  = repeat(meas[t,:],1,np) - repeat(yhat(t)',ny,1)
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

# DRMS convenience wrapper mirroring ekf_tlnn_drms: returns Inf on divergence.
function mpf_online_drms(traj, ins, mag, flux, itp, x0_TL, P0, Qd, R;
                         terms=[:permanent,:induced,:eddy,:bias], num_part=1000,
                         thresh=0.8, roughen_m=0.0,
                         core::Bool=true, warm=600.0, div_thresh=1e4)
    fr = try
        mpf_online(ins,mag,flux,itp,x0_TL,P0,Qd,R;terms=terms,num_part=num_part,
                   thresh=thresh,roughen_m=roughen_m,core=core)
    catch e
        @warn("mpf_online failed",e); return Inf
    end
    fr.c || return Inf
    fo = MagNav.eval_filt(traj,ins,fr)
    m  = (traj.tt .- traj.tt[1]) .>= warm
    dn = dlat2dn.(fo.lat[m] .- traj.lat[m], traj.lat[m])
    de = dlon2de.(fo.lon[m] .- traj.lon[m], traj.lat[m])
    d  = sqrt(mean(dn.^2 .+ de.^2))
    return isfinite(d) && d < div_thresh ? d : Inf
end
