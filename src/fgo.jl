"""
    fgo(lat, lon, alt, vn, ve, vd, fn, fe, fd, Cnb, meas, dt, itp_mapS;
        P0            = create_P0(),
        Qd            = create_Qd(),
        R             = 1.0,
        baro_tau      = 3600.0,
        acc_tau       = 3600.0,
        gyro_tau      = 3600.0,
        fogm_tau      = 600.0,
        date          = get_years(2020,185),
        core::Bool    = false,
        der_mapS      = nothing,
        map_alt       = 0,
        solver::Symbol = :rts,
        robust::Symbol = :none,
        robust_c      = 0,
        q_floor       = 1e-24,
        n_iter        = 5,
        tol           = 1e-4,
        silent        = true)

Factor graph optimization (FGO) for airborne magnetic anomaly navigation.

The navigation problem is posed as a factor graph whose variables are the
error states `x_t` (`t = 1,...,N`) of the Pinson error model (see
[`get_pinson`](@ref)) and whose factors are:

- a **prior factor** on `x_1`, with information `P0^-1`,
- **process (motion) factors** tying `x_{t+1}` to `Φ_t x_t`, with information
  `Qd^-1`, where `Φ_t` is the Pinson transition matrix (see [`get_Phi`](@ref)),
- **measurement factors** on the scalar magnetometer, with information `R^-1`,
  where the expected measurement `h(x_t)` and its Jacobian `H_t` come from the
  magnetic anomaly map (see [`get_h`](@ref) & [`get_H`](@ref)).

The maximum a posteriori (MAP) estimate is the minimizer of the negative
log-likelihood (the factor graph objective)

    J(x_1,...,x_N) = ‖x_1‖²_{P0^-1}
                   + Σ_t ‖x_{t+1} - Φ_t x_t‖²_{Qd^-1}
                   + Σ_t ρ(‖meas_t - h(x_t)‖_{R^-1})

where `ρ` is quadratic by default or a robust kernel (`robust` argument).
Unlike the causal [`ekf`](@ref), every state estimate uses **all** measurements
(past and future), which typically reduces the navigation error, especially
early in the flight.

Two mathematically equivalent solvers are provided:

- `solver = :rts`: iterated fixed-interval (Rauch–Tung–Striebel) smoother,
  i.e., Gauss–Newton on the factor graph chain solved with forward/backward
  recursions. Fast (linear in `N`) and numerically well-behaved even with the
  near-deterministic position process noise.
- `solver = :gn`: "textbook" sparse factor graph solver. All whitened factor
  residuals are stacked into one sparse Jacobian and each Gauss–Newton step is
  solved globally via sparse QR (square-root information form, as in
  square-root SAM/GTSAM). The near-zero process noise entries of `Qd` make the
  information form ill-conditioned, so `q_floor` is added to the `Qd` diagonal
  for whitening (position states have ~1e-30 nominal noise). Final state and
  covariance are extracted with an RTS pass at the converged linearization.

Robust (M-estimator) measurement kernels, per the robust Bayesian inference
approach of Whitney & Nielsen (JNC 2025), are implemented with iteratively
reweighted least squares (IRLS) in both solvers: measurement factor `t` is
reweighted by `w_t = ρ'(e_t)/e_t` of its whitened residual `e_t` between
Gauss–Newton iterations, which suppresses map/measurement outliers (e.g.,
uncharted anomalies, dropouts) that would corrupt a quadratic (Kalman) update.

**Arguments:**
- `lat`:      latitude  [rad]
- `lon`:      longitude [rad]
- `alt`:      altitude  [m]
- `vn`:       north velocity [m/s]
- `ve`:       east  velocity [m/s]
- `vd`:       down  velocity [m/s]
- `fn`:       north specific force [m/s^2]
- `fe`:       east  specific force [m/s^2]
- `fd`:       down  specific force [m/s^2]
- `Cnb`:      direction cosine matrix (body to navigation) [-]
- `meas`:     scalar magnetometer measurement [nT]
- `dt`:       measurement time step [s]
- `itp_mapS`: scalar map interpolation function (`f(lat,lon)` or `f(lat,lon,alt)`)
- `P0`:       (optional) initial covariance matrix
- `Qd`:       (optional) discrete time process/system noise matrix
- `R`:        (optional) measurement (white) noise variance
- `baro_tau`: (optional) barometer time constant [s]
- `acc_tau`:  (optional) accelerometer time constant [s]
- `gyro_tau`: (optional) gyroscope time constant [s]
- `fogm_tau`: (optional) FOGM catch-all time constant [s]
- `date`:     (optional) measurement date (decimal year) for IGRF [yr]
- `core`:     (optional) if true, include core magnetic field in measurement
- `der_mapS`: (optional) scalar map vertical derivative map interpolation function (`f(lat,lon)` or (`f(lat,lon,alt)`)
- `map_alt`:  (optional) map altitude [m]
- `solver`:   (optional) factor graph solver {`:rts`,`:gn`}
- `robust`:   (optional) robust measurement kernel {`:none`,`:huber`,`:cauchy`}
- `robust_c`: (optional) robust kernel tuning constant, `0` for default (`1.345` Huber, `2.385` Cauchy)
- `q_floor`:  (optional) process noise diagonal floor for `:gn` whitening
- `n_iter`:   (optional) maximum number of Gauss–Newton (relinearization/IRLS) iterations
- `tol`:      (optional) convergence tolerance on the RMS smoothed state change between iterations
- `silent`:   (optional) if true, no print outs

**Returns:**
- `filt_res`: `FILTres` filter (smoother) results struct
"""
function fgo(lat, lon, alt, vn, ve, vd, fn, fe, fd, Cnb, meas, dt, itp_mapS;
             P0             = create_P0(),
             Qd             = create_Qd(),
             R              = 1.0,
             baro_tau       = 3600.0,
             acc_tau        = 3600.0,
             gyro_tau       = 3600.0,
             fogm_tau       = 600.0,
             date           = get_years(2020,185),
             core::Bool     = false,
             der_mapS       = nothing,
             map_alt        = 0,
             solver::Symbol = :rts,
             robust::Symbol = :none,
             robust_c       = 0,
             q_floor        = 1e-24,
             n_iter         = 5,
             tol            = 1e-4,
             silent         = true)

    @assert solver in (:rts,:gn)           "solver $solver not defined"
    @assert robust in (:none,:huber,:cauchy) "robust kernel $robust not defined"

    N  = length(lat)
    nx = size(P0,1)
    ny = size(meas,2)

    length(R) == 2 && (R = mean(R)) # adaptive R not supported, use mean
    Rs = mean(R) # scalar measurement variance for whitening & weights

    robust_c == 0 && (robust_c = robust == :cauchy ? 2.385 : 1.345)

    map_cache = itp_mapS isa Map_Cache ? itp_mapS : nothing

    # Pinson transition matrices (t -> t+1), reused by all iterations & solvers
    Phi_a = zeros(eltype(P0),nx,nx,N-1)
    for t = 1:(N-1)
        Phi_a[:,:,t] = get_Phi(nx,lat[t],vn[t],ve[t],vd[t],fn[t],fe[t],fd[t],
                               Cnb[:,:,t],baro_tau,acc_tau,gyro_tau,fogm_tau,dt)
    end

    # per-time-step map interpolation functions (resolve map cache once)
    itps = map_cache isa Map_Cache ?
           [get_cached_map(map_cache,lat[t],lon[t],alt[t];silent=true) for t = 1:N] :
           fill(itp_mapS,N)

    # expected measurement & Jacobian at reference states x_bar [nx x N]
    function meas_model(x_bar)
        h_bar = zeros(eltype(P0),N)
        H_bar = zeros(eltype(P0),nx,N)
        for t = 1:N
            xb = x_bar[:,t]
            if (map_alt > 0) & !(der_mapS isa Nothing)
                h_bar[t] = get_h(itps[t],der_mapS,xb,lat[t],lon[t],alt[t],map_alt;
                                 date=date,core=core)[1]
            else
                h_bar[t] = get_h(itps[t],xb,lat[t],lon[t],alt[t];
                                 date=date,core=core)[1]
            end
            H_bar[:,t] = get_H(itps[t],xb,lat[t],lon[t],alt[t];date=date,core=core)
        end
        return (h_bar, H_bar)
    end

    x_smooth = zeros(eltype(P0),nx,N)
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

        if solver == :gn
            x_smooth = fgo_gn_step(x_bar,h_bar,H_bar,Phi_a,meas,P0,Qd,Rs,w,q_floor)
        else
            (x_smooth,P_smooth) = fgo_rts_pass(x_bar,h_bar,H_bar,Phi_a,meas,
                                               P0,Qd,R,w,ny)
        end

        # convergence check on the RMS change of the smoothed states
        dx = sqrt(mean(abs2, x_smooth .- x_bar))
        if !silent
            J = fgo_cost(x_smooth,Phi_a,meas,P0,Qd,Rs,meas_model)
            @info("fgo ($solver) iter $iter: Δx_rms = $(round(dx,sigdigits=3)), J = $(round(J,sigdigits=6))")
        end
        (iter > 1) && (dx < tol) && break
    end

    # for the global GN solver, extract the covariance with an RTS pass at the
    # converged linearization (states are kept from the sparse QR solution)
    if solver == :gn
        (h_bar,H_bar) = meas_model(x_smooth)
        P_smooth = fgo_rts_pass(x_smooth,h_bar,H_bar,Phi_a,meas,P0,Qd,R,w,ny)[2]
    end

    # post-fit (nonlinear) measurement residuals at the smoothed estimate
    r_out = zeros(eltype(P0),ny,N)
    (h_fit,_) = meas_model(x_smooth)
    for t = 1:N
        r_out[:,t] = meas[t,:] .- h_fit[t]
    end

    return FILTres(x_smooth, P_smooth, r_out, true)
end # function fgo

"""
    fgo(ins::INS, meas, itp_mapS;
        P0             = create_P0(),
        Qd             = create_Qd(),
        R              = 1.0,
        baro_tau       = 3600.0,
        acc_tau        = 3600.0,
        gyro_tau       = 3600.0,
        fogm_tau       = 600.0,
        date           = get_years(2020,185),
        core::Bool     = false,
        der_mapS       = map_itp(zeros(2,2),[-pi,pi],[-pi/2,pi/2]),
        map_alt        = 0,
        solver::Symbol = :rts,
        robust::Symbol = :none,
        robust_c       = 0,
        q_floor        = 1e-24,
        n_iter         = 5,
        tol            = 1e-4,
        silent         = true)

Factor graph optimization (FGO) for airborne magnetic anomaly navigation.

**Arguments:**
- `ins`:      `INS` inertial navigation system struct
- `meas`:     scalar magnetometer measurement [nT]
- `itp_mapS`: scalar map interpolation function (`f(lat,lon)` or `f(lat,lon,alt)`)
- `P0`:       (optional) initial covariance matrix
- `Qd`:       (optional) discrete time process/system noise matrix
- `R`:        (optional) measurement (white) noise variance
- `baro_tau`: (optional) barometer time constant [s]
- `acc_tau`:  (optional) accelerometer time constant [s]
- `gyro_tau`: (optional) gyroscope time constant [s]
- `fogm_tau`: (optional) FOGM catch-all time constant [s]
- `date`:     (optional) measurement date (decimal year) for IGRF [yr]
- `core`:     (optional) if true, include core magnetic field in measurement
- `der_mapS`: (optional) scalar map vertical derivative map interpolation function (`f(lat,lon)` or (`f(lat,lon,alt)`)
- `map_alt`:  (optional) map altitude [m]
- `solver`:   (optional) factor graph solver {`:rts`,`:gn`}
- `robust`:   (optional) robust measurement kernel {`:none`,`:huber`,`:cauchy`}
- `robust_c`: (optional) robust kernel tuning constant, `0` for default (`1.345` Huber, `2.385` Cauchy)
- `q_floor`:  (optional) process noise diagonal floor for `:gn` whitening
- `n_iter`:   (optional) maximum number of Gauss–Newton (relinearization/IRLS) iterations
- `tol`:      (optional) convergence tolerance on the RMS smoothed state change between iterations
- `silent`:   (optional) if true, no print outs

**Returns:**
- `filt_res`: `FILTres` filter (smoother) results struct
"""
function fgo(ins::INS, meas, itp_mapS;
             P0             = create_P0(),
             Qd             = create_Qd(),
             R              = 1.0,
             baro_tau       = 3600.0,
             acc_tau        = 3600.0,
             gyro_tau       = 3600.0,
             fogm_tau       = 600.0,
             date           = get_years(2020,185),
             core::Bool     = false,
             der_mapS       = map_itp(zeros(2,2),[-pi,pi],[-pi/2,pi/2]),
             map_alt        = 0,
             solver::Symbol = :rts,
             robust::Symbol = :none,
             robust_c       = 0,
             q_floor        = 1e-24,
             n_iter         = 5,
             tol            = 1e-4,
             silent         = true)
    fgo(ins.lat,ins.lon,ins.alt,ins.vn,ins.ve,ins.vd,ins.fn,ins.fe,ins.fd,
        ins.Cnb,meas,ins.dt,itp_mapS;
        P0       = P0,
        Qd       = Qd,
        R        = R,
        baro_tau = baro_tau,
        acc_tau  = acc_tau,
        gyro_tau = gyro_tau,
        fogm_tau = fogm_tau,
        date     = date,
        core     = core,
        der_mapS = der_mapS,
        map_alt  = map_alt,
        solver   = solver,
        robust   = robust,
        robust_c = robust_c,
        q_floor  = q_floor,
        n_iter   = n_iter,
        tol      = tol,
        silent   = silent)
end # function fgo

"""
    robust_weight(e, robust::Symbol, c)

Internal helper function to get the IRLS weight `w = ρ'(e)/e` of a robust
kernel `ρ` at whitened residual magnitude `e`.

- `:huber`:  `w = min(1, c/|e|)`
- `:cauchy`: `w = 1 / (1 + (e/c)^2)`
"""
function robust_weight(e, robust::Symbol, c)
    if robust == :huber
        return e <= c ? one(e) : c / abs(e)
    elseif robust == :cauchy
        return 1 / (1 + (e/c)^2)
    else
        return one(e)
    end
end # function robust_weight

"""
    fgo_rts_pass(x_bar, h_bar, H_bar, Phi_a, meas, P0, Qd, R, w, ny;
                 x0 = zeros(eltype(P0),size(x_bar,1)))

Internal helper function for one Gauss–Newton iteration of the factor graph
solved as a fixed-interval (Rauch–Tung–Striebel) smoother: forward filter pass
with the measurement relinearized about `x_bar`, then backward smoothing pass.
Measurement factor `t` is IRLS-reweighted with `w[t]` (i.e., `R/w[t]`).
The prior factor is `x_1 ~ N(x0,P0)`.

**Returns:**
- `x_smooth`: smoothed states
- `P_smooth`: smoothed covariance matrices
"""
function fgo_rts_pass(x_bar, h_bar, H_bar, Phi_a, meas, P0, Qd, R, w, ny;
                      x0 = zeros(eltype(P0),size(x_bar,1)))

    (nx,N) = size(x_bar)

    x_pred = zeros(eltype(P0),nx,N)
    P_pred = zeros(eltype(P0),nx,nx,N)
    x_upd  = zeros(eltype(P0),nx,N)
    P_upd  = zeros(eltype(P0),nx,nx,N)

    # forward pass
    x = collect(eltype(P0),x0)
    P = P0

    for t = 1:N
        x_pred[:,t]   = x
        P_pred[:,:,t] = P

        H = repeat(H_bar[:,t]',ny,1) # ny x nx

        # residual of the relinearized measurement about a priori state x
        resid = meas[t,:] .- (h_bar[t] .+ H*(x .- x_bar[:,t]))

        S = H*P*H' .+ R ./ w[t] # measurement residual covariance (reweighted)
        K = (P*H') / S          # Kalman gain

        x = x + K*resid
        P = (I - K*H) * P
        P = (P + P') / 2        # keep symmetric

        x_upd[:,t]   = x
        P_upd[:,:,t] = P

        if t < N
            Phi = Phi_a[:,:,t]
            x = Phi*x
            P = Phi*P*Phi' + Qd
            P = (P + P') / 2
        end
    end

    # backward (RTS) pass
    x_smooth = zeros(eltype(P0),nx,N)
    P_smooth = zeros(eltype(P0),nx,nx,N)
    x_smooth[:,N]   = x_upd[:,N]
    P_smooth[:,:,N] = P_upd[:,:,N]

    for t = (N-1):-1:1
        Phi = Phi_a[:,:,t]
        C   = (P_upd[:,:,t] * Phi') / P_pred[:,:,t+1] # smoother gain
        x_smooth[:,t]   = x_upd[:,t]   + C*(x_smooth[:,t+1]   - x_pred[:,t+1])
        P_smooth[:,:,t] = P_upd[:,:,t] + C*(P_smooth[:,:,t+1] - P_pred[:,:,t+1])*C'
        P_smooth[:,:,t] = (P_smooth[:,:,t] + P_smooth[:,:,t]') / 2
    end

    # also return the filtered (forward-pass) estimates x_upd/P_upd: for a chain,
    # the filtered state at a node is the marginal of the past sub-graph onto that
    # node, which the fixed-lag window uses as a double-count-free handoff prior.
    return (x_smooth, P_smooth, x_upd, P_upd)
end # function fgo_rts_pass

"""
    fgo_gn_step(x_bar, h_bar, H_bar, Phi_a, meas, P0, Qd, Rs, w, q_floor)

Internal helper function for one global Gauss–Newton step of the factor graph
in square-root information form. All whitened factor residuals (prior, process,
measurement) are stacked into a single sparse Jacobian over the full state
trajectory `X = [x_1; ...; x_N]`, linearized about `x_bar`, and solved with
sparse QR (as in square-root SAM). `q_floor` is added to the `Qd` diagonal
so the near-deterministic position process rows remain representable in the
(inverse) square-root form.

**Returns:**
- `x_new`: updated states [nx x N]
"""
function fgo_gn_step(x_bar, h_bar, H_bar, Phi_a, meas, P0, Qd, Rs, w, q_floor)

    (nx,N) = size(x_bar)
    T  = eltype(P0)
    ny = size(meas,2)

    # whiteners (lower-triangular inverse square-root of factor covariances)
    W0 = inv(chol(P0)')                                    # prior
    Lq = cholesky(Hermitian(collect(Qd) + q_floor*I)).L    # process

    n_rows = nx + (N-1)*nx + N*ny
    n_cols = N*nx
    nnz_ub = nx^2 + (N-1)*2*nx^2 + N*ny*nx

    Is = zeros(Int,nnz_ub)
    Js = zeros(Int,nnz_ub)
    Vs = zeros(T,nnz_ub)
    b  = zeros(T,n_rows)

    k   = 0 # triplet counter
    row = 0 # row counter

    # prior factor rows: W0 * x_1 = 0
    for j = 1:nx, i = 1:nx
        k += 1
        Is[k] = row + i
        Js[k] = j
        Vs[k] = W0[i,j]
    end
    row += nx

    # process factor rows: Lq \ (x_{t+1} - Phi_t x_t) = 0
    for t = 1:(N-1)
        B = Lq \ hcat(-Phi_a[:,:,t], Matrix{T}(I,nx,nx)) # nx x 2nx
        c0 = (t-1)*nx
        for j = 1:(2*nx), i = 1:nx
            k += 1
            Is[k] = row + i
            Js[k] = c0 + j
            Vs[k] = B[i,j]
        end
        row += nx
    end

    # measurement factor rows: sqrt(w/Rs) * (meas - h(x_bar) - H*(x - x_bar)) = 0
    for t = 1:N
        sw = sqrt(w[t]/Rs)
        c0 = (t-1)*nx
        for i = 1:ny
            for j = 1:nx
                k += 1
                Is[k] = row + i
                Js[k] = c0 + j
                Vs[k] = sw*H_bar[j,t]
            end
            b[row+i] = sw*(meas[t,i] - h_bar[t] + dot(H_bar[:,t],x_bar[:,t]))
        end
        row += ny
    end

    A = sparse(Is[1:k],Js[1:k],Vs[1:k],n_rows,n_cols)
    X = qr(A) \ b # global sparse QR (least squares) solve

    return reshape(X,nx,N)
end # function fgo_gn_step

"""
    fgo_cost(x, Phi_a, meas, P0, Qd, Rs, meas_model)

Internal helper function to evaluate the (quadratic) factor graph objective
(negative log-likelihood, up to a constant) at the states `x`:

    J = ‖x_1‖²_{P0^-1}
      + Σ_t ‖x_{t+1} - Φ_t x_t‖²_{Qd^-1}
      + Σ_t ‖meas_t - h(x_t)‖²_{R^-1} .

Used only for reporting optimization convergence.

**Returns:**
- `J`: factor graph objective (scalar)
"""
function fgo_cost(x, Phi_a, meas, P0, Qd, Rs, meas_model)

    N = size(x,2)

    # prior factor on x_1
    J = dot(x[:,1], P0 \ x[:,1])

    # process (motion) factors
    for t = 1:(N-1)
        d  = x[:,t+1] .- Phi_a[:,:,t]*x[:,t]
        J += dot(d, Qd \ d)
    end

    # measurement factors
    (h,_) = meas_model(x)
    for t = 1:N
        J += sum(abs2, meas[t,:] .- h[t]) / Rs
    end

    return (J)
end # function fgo_cost
