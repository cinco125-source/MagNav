"""
    fgo_sensor(lat, lon, alt, vn, ve, vd, fn, fe, fd, Cnb, meas, dt, itp_mapS;
               P0               = create_P0(),
               Qd               = create_Qd(),
               R                = 1.0,
               baro_tau         = 3600.0,
               acc_tau          = 3600.0,
               gyro_tau         = 3600.0,
               fogm_tau         = 600.0,
               date             = get_years(2020,185),
               core::Bool       = false,
               der_mapS         = nothing,
               map_alt          = 0,
               n_harm::Int      = 0,
               heading::Symbol  = :geometry,
               axis             = [0.0,0.0,1.0],
               cal_bias::Bool   = false,
               drift::Bool      = false,
               dead_zone::Bool  = false,
               dz_floor         = 0.05,
               sigma_head       = 50.0,
               sigma_bias       = 50.0,
               sigma_drift      = 1.0,
               q_head           = 1e-4,
               q_bias           = 1e-4,
               q_drift          = 1e-8,
               robust::Symbol   = :none,
               robust_c         = 0,
               n_iter           = 6,
               tol              = 1e-4,
               silent           = true)

Factor graph optimization (FGO) for airborne magnetic anomaly navigation with
explicit, physically-motivated **scalar magnetometer sensor-error factors**.
Extends [`fgo`](@ref) by appending sensor-error states to the Pinson error
states and modeling their contribution to the scalar measurement.

For an optically-pumped / quantum scalar magnetometer, the dominant orientation
errors depend on the angle `θ` between the sensor optical (sensitive) axis `ê`
and the magnetic field `B` (Oelsner et al. 2022; the aircraft heading alone is a
crude proxy). The following sensor-error factors can be enabled individually or
together, all entering the scalar measurement **linearly** (well-conditioned
Gauss–Newton):

- **Heading error** (`n_harm > 0`): the optically-pumped-magnetometer (OPM)
  heading error as a cosine-harmonic expansion in the sensor–field angle `ψ`
  (`= θ`, the angle between the optical axis and the field), following the
  scalar-atomic-magnetometer models of Hager et al. (2026) and Wang et al.
  (2020, IEEE GRSL). The harmonic orders are `{1,2,4}` for `n_harm ≥ 3`:

      Δ_head(t) = Σ_{n∈{1,2,4}} c_n cos(n ψ_t) ,   (linear in the coefficients c_n)

  whose `k=1` term captures the vector light shift and `k=2` the nonlinear
  Zeeman effect. `θ_t = ∠(ê, B_body(t))`, with `B_body = Cnb' B_nav` (IGRF).

- **Fluxgate hard-iron bias** (`cal_bias = true`): a body-fixed vector bias `m`
  projected onto the scalar reading, `Δ_bias(t) = m · û_body(t)` (3 states).

- **Linear drift** (`drift = true`): electronics/sensor drift
  `Δ_drift(t) = d · (t-1) dt` (1 state, nT/s).

- **Dead-zone weighting** (`dead_zone = true`): near the OPM equatorial dead
  zone (field ⟂ optical axis, `|cos ψ| → 0`) the signal amplitude collapses and
  the effective noise is amplified by `1/max(|cos ψ|, ε)` (Hager et al. 2026),
  so the measurement information is weighted by `w = max(|cos ψ|, ε)²`
  (`ε = dz_floor`), down-weighting dead-zone samples — the physical analog of a
  robust kernel.

Unlike a low-dimensional grid/point-mass estimator (position only), the
continuous factor graph estimates these sensor-error states **jointly with
navigation**, recovering a sensor calibration as a by-product. Estimated states
are ordered `[Pinson (nx0) ; heading (n_harm cosine coeffs) ; bias (3) ; drift (1)]`.

**Arguments:** (in addition to those of [`fgo`](@ref))
- `n_harm`:      number of heading-error harmonics (`0` disables the heading factor)
- `heading`:     heading-error basis {`:geometry` (sensor–field angle), `:yaw`}
- `axis`:        sensor optical (sensitive) axis in the body frame [-]
- `cal_bias`:    if true, include the `3`-state fluxgate hard-iron bias factor
- `drift`:       if true, include the `1`-state linear drift factor
- `dead_zone`:   if true, inflate the measurement variance near sensor dead zones
- `dz_floor`:    dead-zone sensitivity floor (avoids ignoring samples entirely)
- `sigma_head`:  heading coefficient prior std dev [nT]
- `sigma_bias`:  hard-iron bias prior std dev [nT]
- `sigma_drift`: drift prior std dev [nT/s]
- `q_head`/`q_bias`/`q_drift`: random-walk driving variances for the sensor states
- `robust`:      robust measurement kernel {`:none`,`:huber`,`:cauchy`}
- `robust_c`:    robust kernel tuning constant, `0` for default
- `n_iter`:      maximum number of Gauss–Newton (relinearization/IRLS) iterations

**Returns:**
- `filt_res`: `FILTres` filter (smoother) results struct (augmented state)
"""
function fgo_sensor(lat, lon, alt, vn, ve, vd, fn, fe, fd, Cnb, meas, dt, itp_mapS;
                    P0               = create_P0(),
                    Qd               = create_Qd(),
                    R                = 1.0,
                    baro_tau         = 3600.0,
                    acc_tau          = 3600.0,
                    gyro_tau         = 3600.0,
                    fogm_tau         = 600.0,
                    date             = get_years(2020,185),
                    core::Bool       = false,
                    der_mapS         = nothing,
                    map_alt          = 0,
                    n_harm::Int      = 0,
                    heading::Symbol  = :geometry,
                    axis             = [0.0,0.0,1.0],
                    cal_bias::Bool   = false,
                    drift::Bool      = false,
                    dead_zone::Bool  = false,
                    dz_floor         = 0.05,
                    sigma_head       = 50.0,
                    sigma_bias       = 50.0,
                    sigma_drift      = 1.0,
                    q_head           = 1e-4,
                    q_bias           = 1e-4,
                    q_drift          = 1e-8,
                    robust::Symbol   = :none,
                    robust_c         = 0,
                    n_iter           = 6,
                    tol              = 1e-4,
                    silent           = true)

    @assert robust  in (:none,:huber,:cauchy) "robust kernel $robust not defined"
    @assert heading in (:geometry,:yaw)       "heading basis $heading not defined"

    N   = length(lat)
    nx0 = size(P0,1)
    # OPM heading error uses cosine harmonics at orders {1,2,4} (Hager et al.
    # 2026; Wang et al. 2020): delta_h = sum_n c_n cos(n psi). One coefficient
    # per harmonic (not a full Fourier cos/sin pair).
    harm_orders = n_harm >= 3 ? [1,2,4] : collect(1:n_harm)
    nh  = length(harm_orders)
    nb  = cal_bias ? 3 : 0
    nd  = drift    ? 1 : 0
    nx  = nx0 + nh + nb + nd
    ny  = size(meas,2)

    length(R) == 2 && (R = mean(R))
    Rs = mean(R)
    robust_c == 0 && (robust_c = robust == :cauchy ? 2.385 : 1.345)

    map_cache = itp_mapS isa Map_Cache ? itp_mapS : nothing
    itps = map_cache isa Map_Cache ?
           [get_cached_map(map_cache,lat[t],lon[t],alt[t];silent=true) for t = 1:N] :
           fill(itp_mapS,N)

    # sensor-field geometry: unit field direction (body frame) & angle to axis
    ax     = axis ./ norm(axis)
    u_body = zeros(eltype(P0),3,N) # unit core-field direction in body frame
    theta  = zeros(eltype(P0),N)   # angle between optical axis & field [rad]
    for t = 1:N
        v = igrf(date,alt[t],lat[t],lon[t],Val(:geodetic))
        u_body[:,t] = Cnb[:,:,t]' * (v ./ norm(v))
        theta[t]    = acos(clamp(dot(ax,u_body[:,t]),-1,1))
    end

    # heading harmonic angle (geometry angle θ or aircraft yaw ψ)
    if heading == :yaw
        (_,_,ang) = dcm2euler(Cnb,:body2nav)
        ang = vec([ang;])
    else
        ang = theta
    end

    # sensor-error measurement bases (each row is that factor's Jacobian entry)
    # OPM heading error: cosine harmonics at orders {1,2,4} (Hager 2026 eq. 20).
    Bh = zeros(eltype(P0),nh,N)
    for (i,n) in enumerate(harm_orders)
        Bh[i,:] = cos.(n .* ang)
    end
    Bb = cal_bias ? u_body : zeros(eltype(P0),0,N)          # hard-iron bias
    Bd = drift ? reshape((0:N-1) .* dt,1,N) .* one(eltype(P0)) :
                 zeros(eltype(P0),0,N)                       # linear drift [s]

    # dead-zone measurement weight (OPM equatorial dead zone, Hager 2026 eq. 21):
    # the signal amplitude collapses as the field nears perpendicular to the
    # optical axis (|cos psi| -> 0), amplifying the effective noise by
    # 1/max(|cos psi|, eps); hence the measurement information is weighted by
    # w = max(|cos psi|, eps)^2. eps = dz_floor regularizes the loss-of-lock zone.
    w_dz = ones(eltype(P0),N)
    if dead_zone
        for t = 1:N
            w_dz[t] = max(abs(cos(theta[t])), dz_floor)^2
        end
    end

    # augmented prior covariance & process noise
    P0_a = zeros(eltype(P0),nx,nx); P0_a[1:nx0,1:nx0] = P0
    Qd_a = zeros(eltype(P0),nx,nx); Qd_a[1:nx0,1:nx0] = Qd
    seti = nx0
    for _ = 1:nh; seti += 1; P0_a[seti,seti] = sigma_head^2;  Qd_a[seti,seti] = q_head;  end
    for _ = 1:nb; seti += 1; P0_a[seti,seti] = sigma_bias^2;  Qd_a[seti,seti] = q_bias;  end
    for _ = 1:nd; seti += 1; P0_a[seti,seti] = sigma_drift^2; Qd_a[seti,seti] = q_drift; end

    # augmented transition: Pinson block + identity (constant) sensor block
    Phi_a = zeros(eltype(P0),nx,nx,N-1)
    for t = 1:(N-1)
        Phi_a[1:nx0,1:nx0,t] = get_Phi(nx0,lat[t],vn[t],ve[t],vd[t],fn[t],fe[t],
                                       fd[t],Cnb[:,:,t],baro_tau,acc_tau,gyro_tau,
                                       fogm_tau,dt)
        for i = 1:(nh+nb+nd)
            Phi_a[nx0+i,nx0+i,t] = 1
        end
    end

    ih = nx0                # heading states start after this index
    ib = nx0 + nh           # bias    states start after this index
    id = nx0 + nh + nb      # drift   states start after this index

    # expected measurement & Jacobian at reference states x_bar [nx x N]
    function meas_model(x_bar)
        h_bar = zeros(eltype(P0),N)
        H_bar = zeros(eltype(P0),nx,N)
        for t = 1:N
            x18 = x_bar[1:nx0,t]
            if (map_alt > 0) & !(der_mapS isa Nothing)
                h_map = get_h(itps[t],der_mapS,x18,lat[t],lon[t],alt[t],map_alt;
                              date=date,core=core)[1]
            else
                h_map = get_h(itps[t],x18,lat[t],lon[t],alt[t];date=date,core=core)[1]
            end
            H_map = get_H(itps[t],x18,lat[t],lon[t],alt[t];date=date,core=core)

            h_head  = nh > 0 ? dot(Bh[:,t],x_bar[ih+1:ih+nh,t]) : zero(eltype(P0))
            h_bias  = nb > 0 ? dot(Bb[:,t],x_bar[ib+1:ib+nb,t]) : zero(eltype(P0))
            h_drift = nd > 0 ? Bd[1,t]*x_bar[id+1,t]            : zero(eltype(P0))

            h_bar[t]   = h_map + h_head + h_bias + h_drift
            H_bar[:,t] = vcat(H_map, Bh[:,t], Bb[:,t], Bd[:,t])
        end
        return (h_bar, H_bar)
    end

    x_smooth = zeros(eltype(P0),nx,N)
    P_smooth = zeros(eltype(P0),nx,nx,N)
    w        = copy(w_dz)

    for iter = 1:n_iter
        x_bar = copy(x_smooth)
        (h_bar,H_bar) = meas_model(x_bar)

        if (robust != :none) & (iter > 1)
            for t = 1:N
                e    = abs(mean(meas[t,:]) - h_bar[t]) / sqrt(Rs)
                w[t] = w_dz[t] * clamp(robust_weight(e,robust,robust_c),1e-6,1)
            end
        end

        (x_smooth,P_smooth) = fgo_rts_pass(x_bar,h_bar,H_bar,Phi_a,meas,
                                           P0_a,Qd_a,R,w,ny)

        dx = sqrt(mean(abs2, x_smooth .- x_bar))
        silent || @info("fgo_sensor iter $iter: Δx_rms = $(round(dx,sigdigits=3))")
        (iter > 1) && (dx < tol) && break
    end

    r_out = zeros(eltype(P0),ny,N)
    (h_fit,_) = meas_model(x_smooth)
    for t = 1:N
        r_out[:,t] = meas[t,:] .- h_fit[t]
    end

    return FILTres(x_smooth, P_smooth, r_out, true)
end # function fgo_sensor

"""
    fgo_sensor(ins::INS, meas, itp_mapS; kwargs...)

Factor graph optimization (FGO) with scalar magnetometer sensor-error factors.

**Arguments:**
- `ins`:      `INS` inertial navigation system struct
- `meas`:     scalar magnetometer measurement [nT]
- `itp_mapS`: scalar map interpolation function (`f(lat,lon)` or `f(lat,lon,alt)`)
- `kwargs`:   see the array method of [`fgo_sensor`](@ref)

**Returns:**
- `filt_res`: `FILTres` filter (smoother) results struct (augmented state)
"""
function fgo_sensor(ins::INS, meas, itp_mapS; kwargs...)
    fgo_sensor(ins.lat,ins.lon,ins.alt,ins.vn,ins.ve,ins.vd,ins.fn,ins.fe,ins.fd,
               ins.Cnb,meas,ins.dt,itp_mapS; kwargs...)
end # function fgo_sensor
