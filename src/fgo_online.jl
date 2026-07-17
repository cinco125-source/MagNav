"""
    fgo_online(lat, lon, alt, vn, ve, vd, fn, fe, fd, Cnb, meas,
               Bx, By, Bz, dt, itp_mapS, x0_TL, P0, Qd, R;
               baro_tau       = 3600.0,
               acc_tau        = 3600.0,
               gyro_tau       = 3600.0,
               fogm_tau       = 600.0,
               date           = get_years(2020,185),
               core::Bool     = false,
               terms          = [:permanent,:induced,:eddy,:bias],
               Bt_scale       = 50000,
               robust::Symbol = :none,
               robust_c       = 0,
               n_iter         = 5,
               tol            = 1e-4,
               silent         = true)

Factor graph optimization (FGO) with batch estimation of Tolles-Lawson
coefficients (aeromagnetic compensation states).

The Tolles-Lawson coefficients are appended to the Pinson error states as
variables of the factor graph (compensation factors), mirroring the state
augmentation of [`ekf_online`](@ref), and the measurement factor becomes

    meas_t = A_t' x_TL + map(lat_t,lon_t,alt_t) + S_t (+ IGRF core)

where `A_t` is the Tolles-Lawson matrix row from the vector magnetometer (see
[`create_TL_A`](@ref)) and `S_t` is the FOGM catch-all state. A prior factor
centered on `x0_TL` (with covariance from `P0`) anchors the coefficients, and
the process factors (with driving noise from `Qd`, i.e., `TL_sigma`) allow
them to vary slowly in flight. Unlike the sequential [`ekf_online`](@ref),
the batch solution estimates the compensation jointly over the **whole**
flight, so early navigation states benefit from calibration information that
only becomes observable later (e.g., after maneuvers).

The maximum a posteriori (MAP) estimate is computed with an iterated
fixed-interval (Rauch–Tung–Striebel) smoother, i.e., Gauss–Newton on the
factor graph chain (see [`fgo`](@ref)). Optional robust (M-estimator)
measurement kernels are applied with iteratively reweighted least squares.

**Arguments:**
- `lat`:          latitude  [rad]
- `lon`:          longitude [rad]
- `alt`:          altitude  [m]
- `vn`:           north velocity [m/s]
- `ve`:           east  velocity [m/s]
- `vd`:           down  velocity [m/s]
- `fn`:           north specific force [m/s^2]
- `fe`:           east  specific force [m/s^2]
- `fd`:           down  specific force [m/s^2]
- `Cnb`:          direction cosine matrix (body to navigation) [-]
- `meas`:         scalar magnetometer measurement [nT]
- `Bx`,`By`,`Bz`: vector magnetometer measurements [nT]
- `dt`:           measurement time step [s]
- `itp_mapS`:     scalar map interpolation function (`f(lat,lon)` or `f(lat,lon,alt)`)
- `x0_TL`:        initial Tolles-Lawson coefficient states
- `P0`:           initial covariance matrix
- `Qd`:           discrete time process/system noise matrix
- `R`:            measurement (white) noise variance
- `baro_tau`:     (optional) barometer time constant [s]
- `acc_tau`:      (optional) accelerometer time constant [s]
- `gyro_tau`:     (optional) gyroscope time constant [s]
- `fogm_tau`:     (optional) FOGM catch-all time constant [s]
- `date`:         (optional) measurement date (decimal year) for IGRF [yr]
- `core`:         (optional) if true, include core magnetic field in measurement
- `terms`:        (optional) Tolles-Lawson terms to use {`:permanent`,`:induced`,`:eddy`,`:bias`}
- `Bt_scale`:     (optional) scaling factor for induced & eddy current terms [nT]
- `robust`:       (optional) robust measurement kernel {`:none`,`:huber`,`:cauchy`}
- `robust_c`:     (optional) robust kernel tuning constant, `0` for default (`1.345` Huber, `2.385` Cauchy)
- `win`:          (optional) fixed-lag window length [s], `0` for a single full-batch fit
- `overlap`:      (optional) window overlap [s] used as warm-up and discarded
- `n_iter`:       (optional) maximum number of Gauss–Newton (relinearization/IRLS) iterations
- `tol`:          (optional) convergence tolerance on the RMS smoothed state change between iterations
- `silent`:       (optional) if true, no print outs

**Returns:**
- `filt_res`: `FILTres` filter (smoother) results struct
"""
function fgo_online(lat, lon, alt, vn, ve, vd, fn, fe, fd, Cnb, meas,
                    Bx, By, Bz, dt, itp_mapS, x0_TL, P0, Qd, R;
                    baro_tau       = 3600.0,
                    acc_tau        = 3600.0,
                    gyro_tau       = 3600.0,
                    fogm_tau       = 600.0,
                    date           = get_years(2020,185),
                    core::Bool     = false,
                    terms          = [:permanent,:induced,:eddy,:bias],
                    Bt_scale       = 50000,
                    robust::Symbol = :none,
                    robust_c       = 0,
                    win            = 0.0,
                    overlap        = 0.0,
                    x0_prior       = nothing,
                    obs_gate::Bool = false,
                    obs_gate_thresh = 0.5,
                    obs_gate_min   = 0.05,
                    A_extra        = nothing,
                    n_iter         = 5,
                    tol            = 1e-4,
                    silent         = true)

    @assert robust in (:none,:huber,:cauchy) "robust kernel $robust not defined"

    # fixed-lag / sliding-window smoother (incremental FGO, iSAM2-style): let the
    # Tolles-Lawson calibration adapt over a long flight instead of one static
    # batch fit. Each window is a batch FGO whose TL estimate is carried forward.
    if win > 0
        return fgo_online_window(lat,lon,alt,vn,ve,vd,fn,fe,fd,Cnb,meas,Bx,By,Bz,
                                 dt,itp_mapS,x0_TL,P0,Qd,R;
                                 win=win,overlap=overlap,baro_tau=baro_tau,
                                 acc_tau=acc_tau,gyro_tau=gyro_tau,fogm_tau=fogm_tau,
                                 date=date,core=core,terms=terms,Bt_scale=Bt_scale,
                                 robust=robust,robust_c=robust_c,obs_gate=obs_gate,
                                 obs_gate_thresh=obs_gate_thresh,obs_gate_min=obs_gate_min,
                                 A_extra=A_extra,n_iter=n_iter,tol=tol,silent=silent)
    end

    N      = length(lat)
    ny     = size(meas,2)
    nx     = size(P0,1)
    nx_TL  = length(x0_TL)
    nx_vec = nx - 18 - nx_TL

    @assert nx_vec == 0 "vector magnetometer states not supported for fgo_online"

    length(R) == 2 && (R = mean(R)) # adaptive R not supported, use mean
    Rs = mean(R)

    robust_c == 0 && (robust_c = robust == :cauchy ? 2.385 : 1.345)

    A = create_TL_A(Bx,By,Bz;
                    terms    = terms,
                    Bt_scale = Bt_scale)
    # optional extra compensation columns (e.g. an endogenous feature such as the
    # scalar mag itself) to study/exercise the observability gate
    A_extra === nothing || (A = hcat(A, collect(eltype(A),A_extra)))
    @assert size(A,2) == nx_TL "x0_TL length ($nx_TL) must match TL columns ($(size(A,2)))"

    map_cache = itp_mapS isa Map_Cache ? itp_mapS : nothing

    # Pinson transition matrices (t -> t+1); TL states propagate as identity
    Phi_a = zeros(eltype(P0),nx,nx,N-1)
    for t = 1:(N-1)
        Phi_a[:,:,t] = get_Phi(nx,lat[t],vn[t],ve[t],vd[t],fn[t],fe[t],fd[t],
                               Cnb[:,:,t],baro_tau,acc_tau,gyro_tau,fogm_tau,dt)
    end

    # per-time-step map interpolation functions (resolve map cache once)
    itps = map_cache isa Map_Cache ?
           [get_cached_map(map_cache,lat[t],lon[t],alt[t];silent=true) for t = 1:N] :
           fill(itp_mapS,N)

    # prior mean: zero Pinson errors, x0_TL compensation coefficients (or a full
    # carried-forward state when x0_prior is supplied, e.g., by the window smoother)
    x0 = zeros(eltype(P0),nx)
    x0[18:17+nx_TL] = x0_TL
    x0_prior === nothing || (x0 = collect(eltype(P0),x0_prior))

    # expected measurement & Jacobian at reference states x_bar [nx x N]
    # h(x) = A_t' x_TL + map(pos) + S (+ core); Jacobian per ekf_online
    function meas_model(x_bar)
        h_bar = zeros(eltype(P0),N)
        H_bar = zeros(eltype(P0),nx,N)
        for t = 1:N
            xb   = x_bar[:,t]
            x_TL = xb[18:17+nx_TL]
            h_bar[t] = (A[t,:]'*x_TL .+
                        get_h(itps[t],xb,lat[t],lon[t],alt[t];
                              date=date,core=core))[1]
            Hll = get_H(itps[t],xb,lat[t],lon[t],alt[t];date=date,core=core)
            H_bar[:,t] = [Hll[1:2]; zeros(eltype(P0),nx-3-nx_TL); A[t,:]; 1]
        end
        return (h_bar, H_bar)
    end

    # FGO-native observability gate. The factor-graph window exposes the full
    # measurement Jacobian, so we can measure the position↔TL confound directly:
    # ρ_obs = how well the TL basis (H rows 18:17+nx_TL, i.e. A) can reproduce the
    # POSITION sensitivity (H rows 1:2, the linearized map gradient ∂h/∂pos) over
    # the window. ρ_obs → 1 means the compensation can mimic a position error ⇒
    # observability collapse. When ρ_obs exceeds a threshold we stiffen the TL
    # prior/process-noise so the calibration cannot chase the map (a causal EKF
    # cannot see this — it has no window information matrix). Default: off.
    Pg = P0; Qg = Qd
    if obs_gate || !silent
        (_,H0) = meas_model(repeat(x0,1,N))
        ρ_obs  = obs_collapse_index(H0, nx_TL)
        gated  = obs_gate && (ρ_obs > obs_gate_thresh)
        if gated
            g  = clamp(one(eltype(P0)) - ρ_obs, obs_gate_min, one(eltype(P0)))
            tl = 18:17+nx_TL
            Pg = copy(P0); Pg[tl,tl] .*= g
            Qg = copy(Qd); Qg[tl,tl] .*= g
        end
        silent || @info("fgo_online obs: ρ_obs=$(round(ρ_obs,digits=3))"*
                        (gated ? " (gated)" : ""))
    end

    x_smooth = repeat(x0,1,N) # initial linearization reference
    P_smooth = zeros(eltype(P0),nx,nx,N)
    w        = ones(eltype(P0),N) # IRLS measurement weights

    for iter = 1:n_iter

        x_bar = copy(x_smooth) # relinearize about previous solution
        (h_bar,H_bar) = meas_model(x_bar)

        # IRLS robust weight update from whitened residuals at x_bar
        if (robust != :none) & (iter > 1)
            for t = 1:N
                e    = abs(mean(meas[t,:]) - h_bar[t]) / sqrt(Rs)
                w[t] = clamp(robust_weight(e,robust,robust_c),1e-6,1)
            end
        end

        (x_smooth,P_smooth) = fgo_rts_pass(x_bar,h_bar,H_bar,Phi_a,meas,
                                           Pg,Qg,R,w,ny;x0=x0)

        # convergence check on the RMS change of the smoothed states
        dx = sqrt(mean(abs2, x_smooth .- x_bar))
        silent || @info("fgo_online iter $iter: Δx_rms = $(round(dx,sigdigits=3))")
        (iter > 1) && (dx < tol) && break
    end

    # post-fit (nonlinear) measurement residuals at the smoothed estimate
    r_out = zeros(eltype(P0),ny,N)
    (h_fit,_) = meas_model(x_smooth)
    for t = 1:N
        r_out[:,t] = meas[t,:] .- h_fit[t]
    end

    return FILTres(x_smooth, P_smooth, r_out, true)
end # function fgo_online

"""
    fgo_online_window(lat, ..., R; win, overlap, kwargs...)

Internal helper: fixed-lag / sliding-window `fgo_online`. The flight is processed
in overlapping windows of length `win` [s] (overlap `overlap` [s]); each window is
a batch FGO whose Tolles-Lawson estimate (and its covariance) is carried forward
as the prior for the next window, so the calibration adapts to time-varying
platform interference (the FGO analog of an online/adaptive filter, à la iSAM2).
Each window commits only its leading stride (window minus overlap); the trailing
overlap serves as smoother look-ahead for the committed epochs and is re-processed
by the next window under the carried prior (bounded double-counting of the overlap
information; the committed means are unaffected).

**Returns:**
- `filt_res`: `FILTres` stitched filter (smoother) results struct
"""
function fgo_online_window(lat, lon, alt, vn, ve, vd, fn, fe, fd, Cnb, meas,
                           Bx, By, Bz, dt, itp_mapS, x0_TL, P0, Qd, R;
                           win, overlap, A_extra = nothing, kwargs...)
    N     = length(lat)
    ny    = size(meas,2)
    nx    = size(P0,1)
    nx_TL = length(x0_TL)
    Lw    = max(2, round(Int, win/dt))
    Lo    = clamp(round(Int, overlap/dt), 0, Lw-1)
    stride = max(1, Lw - Lo)

    x_out = zeros(eltype(P0),nx,N)
    P_out = zeros(eltype(P0),nx,nx,N)
    r_out = zeros(eltype(P0),ny,N)

    x0_pr = nothing                 # full carried-forward prior mean (nothing = fresh)
    P0_c  = copy(P0)
    i0    = 1

    # Each window covers [i0, i0+Lw-1] but only COMMITS its leading `stride`
    # samples [i0, i0+stride-1]; the trailing `overlap` samples act as smoother
    # look-ahead (future context) and are re-committed by the next window. This
    # tiles the flight contiguously (no gaps). The full state (navigation error +
    # TL calibration) is carried forward, so the calibration adapts over the
    # flight while navigation stays continuous across windows.
    while i0 <= N
        i1 = min(i0+Lw-1, N)
        S  = i0:i1
        # extra compensation columns are per-epoch rows: slice them to the window
        Ax = A_extra === nothing ? nothing :
             (A_extra isa AbstractVector ? A_extra[S] : A_extra[S,:])
        res = fgo_online(lat[S],lon[S],alt[S],vn[S],ve[S],vd[S],fn[S],fe[S],fd[S],
                         Cnb[:,:,S],meas[S,:],Bx[S],By[S],Bz[S],dt,itp_mapS,
                         x0_TL,P0_c,Qd,R; win=0.0,overlap=0.0,x0_prior=x0_pr,
                         A_extra=Ax, kwargs...)

        gc1 = (i1==N) ? N : min(i0+stride-1, N)   # committed global end
        lc1 = gc1-i0+1                            # local index of commit end
        x_out[:,i0:gc1]   = res.x[:,1:lc1]
        P_out[:,:,i0:gc1] = res.P[:,:,1:lc1]
        r_out[:,i0:gc1]   = res.r[:,1:lc1]

        # carry the full smoothed state (mean + covariance) at the commit boundary
        x0_pr = res.x[:,lc1]
        P0_c  = (res.P[:,:,lc1] .+ res.P[:,:,lc1]') ./ 2   # keep symmetric

        i1 == N && break
        i0 += stride
    end

    return FILTres(x_out, P_out, r_out, true)
end # function fgo_online_window

"""
    fgo_online(ins::INS, meas, flux::MagV, itp_mapS, x0_TL, P0, Qd, R;
               baro_tau       = 3600.0,
               acc_tau        = 3600.0,
               gyro_tau       = 3600.0,
               fogm_tau       = 600.0,
               date           = get_years(2020,185),
               core::Bool     = false,
               terms          = [:permanent,:induced,:eddy,:bias],
               Bt_scale       = 50000,
               robust::Symbol = :none,
               robust_c       = 0,
               n_iter         = 5,
               tol            = 1e-4,
               silent         = true)

Factor graph optimization (FGO) with batch estimation of Tolles-Lawson
coefficients (aeromagnetic compensation states).

**Arguments:**
- `ins`:      `INS` inertial navigation system struct
- `meas`:     scalar magnetometer measurement [nT]
- `flux`:     `MagV` vector magnetometer measurement struct
- `itp_mapS`: scalar map interpolation function (`f(lat,lon)` or `f(lat,lon,alt)`)
- `x0_TL`:    initial Tolles-Lawson coefficient states
- `P0`:       initial covariance matrix
- `Qd`:       discrete time process/system noise matrix
- `R`:        measurement (white) noise variance
- `baro_tau`: (optional) barometer time constant [s]
- `acc_tau`:  (optional) accelerometer time constant [s]
- `gyro_tau`: (optional) gyroscope time constant [s]
- `fogm_tau`: (optional) FOGM catch-all time constant [s]
- `date`:     (optional) measurement date (decimal year) for IGRF [yr]
- `core`:     (optional) if true, include core magnetic field in measurement
- `terms`:    (optional) Tolles-Lawson terms to use {`:permanent`,`:induced`,`:eddy`,`:bias`}
- `Bt_scale`: (optional) scaling factor for induced & eddy current terms [nT]
- `robust`:   (optional) robust measurement kernel {`:none`,`:huber`,`:cauchy`}
- `robust_c`: (optional) robust kernel tuning constant, `0` for default (`1.345` Huber, `2.385` Cauchy)
- `n_iter`:   (optional) maximum number of Gauss–Newton (relinearization/IRLS) iterations
- `tol`:      (optional) convergence tolerance on the RMS smoothed state change between iterations
- `silent`:   (optional) if true, no print outs

**Returns:**
- `filt_res`: `FILTres` filter (smoother) results struct
"""
function fgo_online(ins::INS, meas, flux::MagV, itp_mapS, x0_TL, P0, Qd, R;
                    baro_tau       = 3600.0,
                    acc_tau       = 3600.0,
                    gyro_tau       = 3600.0,
                    fogm_tau       = 600.0,
                    date           = get_years(2020,185),
                    core::Bool     = false,
                    terms          = [:permanent,:induced,:eddy,:bias],
                    Bt_scale       = 50000,
                    robust::Symbol = :none,
                    robust_c       = 0,
                    win            = 0.0,
                    overlap        = 0.0,
                    obs_gate::Bool = false,
                    obs_gate_thresh = 0.5,
                    obs_gate_min   = 0.05,
                    A_extra        = nothing,
                    n_iter         = 5,
                    tol            = 1e-4,
                    silent         = true)
    fgo_online(ins.lat,ins.lon,ins.alt,ins.vn,ins.ve,ins.vd,ins.fn,ins.fe,ins.fd,
               ins.Cnb,meas,flux.x,flux.y,flux.z,ins.dt,itp_mapS,x0_TL,P0,Qd,R;
               baro_tau = baro_tau,
               acc_tau  = acc_tau,
               gyro_tau = gyro_tau,
               fogm_tau = fogm_tau,
               date     = date,
               core     = core,
               terms    = terms,
               Bt_scale = Bt_scale,
               robust   = robust,
               robust_c = robust_c,
               win      = win,
               overlap  = overlap,
               obs_gate = obs_gate,
               obs_gate_thresh = obs_gate_thresh,
               obs_gate_min    = obs_gate_min,
               A_extra  = A_extra,
               n_iter   = n_iter,
               tol      = tol,
               silent   = silent)
end # function fgo_online

"""
    obs_collapse_index(H, nx_TL)

Internal helper: FGO-native observability collapse index for joint aeromagnetic
compensation + navigation. Given the stacked measurement Jacobian `H` (`nx` x `N`)
of a factor-graph window, returns how well the Tolles-Lawson basis (rows
`18:17+nx_TL`, i.e. the `A`-matrix) can linearly reproduce the POSITION
sensitivity (rows `1:2`, the map gradient ∂h/∂pos) over the window — the max
`R²` across the latitude/longitude directions. `→ 1` means the compensation can
mimic a position error, i.e. the joint estimate is unobservable (collapse); `≈ 0`
means the compensation is orthogonal to the navigation signal (observable).
"""
function obs_collapse_index(H, nx_TL)
    N = size(H,2)
    T = permutedims(H[18:17+nx_TL, :])          # N × nx_TL (TL basis rows = A)
    A = [ones(eltype(H),N) T]
    ρ = zero(eltype(H))
    for j in 1:2                                 # lat, lon position sensitivities
        y   = H[j, :]
        yc  = y .- mean(y); sst = sum(abs2, yc)
        sst == 0 && continue
        β   = A \ y
        ρj  = 1 - sum(abs2, y .- A*β)/sst
        ρ   = max(ρ, ρj)
    end
    return ρ
end # function obs_collapse_index
