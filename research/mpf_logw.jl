##* Native MagNav mpf with a LOG-DOMAIN (max-shifted) weight update.
#
# The ONLY change from src/mpf.jl is the particle-weight computation: the stock mpf
# forms weights as raw exp(-0.5 e^2/V), so a step where every particle has a large
# innovation underflows all weights to 0 and the filter is wrongly declared
# "diverged". Doing the weights in the log domain with a max-shift (only relative
# likelihoods matter) removes that numerical failure. Pure map-matching, no online
# TL — the clean test of whether the RBPF itself works on our full lines.

using MagNav
using LinearAlgebra, Statistics
using Random: randn

function mpf_logw(ins::MagNav.INS, meas, itp_mapS;
                 P0, Qd, R, num_part=1000, thresh=0.8, roughen_m=0.0,
                 baro_tau=3600.0, acc_tau=3600.0, gyro_tau=3600.0, fogm_tau=600.0,
                 date=MagNav.get_years(2020,185), core::Bool=false)
    lat=ins.lat; lon=ins.lon; alt=ins.alt; vn=ins.vn; ve=ins.ve; vd=ins.vd
    fn=ins.fn; fe=ins.fe; fd=ins.fd; Cnb=ins.Cnb; dt=ins.dt
    N=length(lat); np=num_part; nx=size(P0,1); nxn=2; nxl=nx-nxn
    meas = reshape(meas, :, 1); ny=1; T2=eltype(P0)

    H=[zeros(T2,1,nxl-1) 1]
    x_out=zeros(T2,nx,N); Pn_out=zeros(T2,nxn,nxn,N); Pl_out=zeros(T2,nxl,nxl,N)
    resid=zeros(T2,ny,N)
    xn=MagNav.chol(P0[1:nxn,1:nxn])'*randn(T2,nxn,np)
    xl=zeros(T2,nxl,np); Pl=P0[nxn+1:end,nxn+1:end]; q=ones(T2,np)/np

    for t=1:N
        Phi=MagNav.get_Phi(nx,lat[t],vn[t],ve[t],vd[t],fn[t],fe[t],fd[t],Cnb[:,:,t],
                           baro_tau,acc_tau,gyro_tau,fogm_tau,dt)
        An_l=Phi[1:nxn,nxn+1:end]; An_n=Phi[1:nxn,1:nxn]
        Al_l=Phi[nxn+1:end,nxn+1:end]; Al_n=Phi[nxn+1:end,1:nxn]

        y_hat=MagNav.get_h(itp_mapS,[xn;xl],lat[t],lon[t],alt[t];date=date,core=core)
        e=repeat(meas[t,:],1,np)-repeat(y_hat',ny,1)
        resid[:,t]=mean(e,dims=2)
        V=H*Pl*H'.+R
        # --- log-domain weight update (the only change vs src/mpf.jl) ---
        logw=zeros(T2,np)
        for i=1:ny; logw.+=-0.5.*(e[i,:].*(1/V[i,i]).*e[i,:]); end
        q=q.*exp.(logw.-maximum(logw))
        if sum(q)>eps(T2)
            q=q/sum(q)
            for i=1:nxn; x_out[i,t]=sum(q.*xn[i,:]); end
            Pn_out[:,:,t]=MagNav.part_cov(q,xn,x_out[1:nxn,t])
            if 1/sum(q.^2)<np*thresh
                ind=MagNav.sys_resample(q); xn=xn[:,ind]; xl=xl[:,ind]; q=ones(T2,np)/np
                if roughen_m > 0   # regularized-PF roughening: reinject position diversity
                    xn[1,:] .+= MagNav.dn2dlat(roughen_m, lat[t]).*randn(T2,np)
                    xn[2,:] .+= MagNav.de2dlon(roughen_m, lat[t]).*randn(T2,np)
                end
            end
        else
            return MagNav.FILTres(x_out, MagNav.filter_exit(Pl_out,Pn_out,t,false), resid, false)
        end
        K=Pl*H'/V; Pl=Pl-K*V*K'
        y_hat=MagNav.get_h(itp_mapS,[xn;xl],lat[t],lon[t],alt[t];date=date,core=core)
        e=repeat(meas[t,:],1,np)-repeat(y_hat',ny,1)
        xl_temp=xl; xl=xl+K*e
        for i=1:nxl; x_out[nxn+i,t]=sum(q.*xl[i,:]); end
        Pl_out[:,:,t]=MagNav.part_cov(q,xl,x_out[1+nxn:end,t],Pl)
        M=An_l*Pl*An_l'+Qd[1:nxn,1:nxn]; L=Al_l*Pl*An_l'/M
        Pl=Al_l*Pl*Al_l'+Qd[nxn+1:end,nxn+1:end]-L*M*L'; Pl=(Pl+Pl')/2
        xn_temp=xn; xn=An_n*xn_temp+An_l*xl_temp+MagNav.chol(M)'*randn(T2,nxn,np)
        z=xn-An_n*xn_temp; xl=Al_n*xn_temp+Al_l*xl+L*(z-An_l*xl)
    end
    return MagNav.FILTres(x_out, MagNav.filter_exit(Pl_out,Pn_out,N,true), resid, true)
end
