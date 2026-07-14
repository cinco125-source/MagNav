using MagNav, Test, MAT
using LinearAlgebra
using Statistics: mean
using MagNav: FILTres

test_file = joinpath(@__DIR__,"test_data","test_data_ekf.mat")
ekf_data  = matopen(test_file,"r") do file
    read(file,"ekf_data")
end

test_file = joinpath(@__DIR__,"test_data","test_data_ins.mat")
ins_data  = matopen(test_file,"r") do file
    read(file,"ins_data")
end

test_file = joinpath(@__DIR__,"test_data","test_data_map.mat")
map_data  = matopen(test_file,"r") do file
    read(file,"map_data")
end

test_file = joinpath(@__DIR__,"test_data","test_data_params.mat")
params    = matopen(test_file,"r") do file
    read(file,"params")
end

test_file = joinpath(@__DIR__,"test_data","test_data_traj.mat")
traj_data = matopen(test_file,"r") do file
    read(file,"traj")
end

P0 = ekf_data["P0"]
Qd = ekf_data["Qd"]
R  = ekf_data["R"]

ins_lat  = deg2rad.(vec(ins_data["lat"]))
ins_lon  = deg2rad.(vec(ins_data["lon"]))
ins_alt  = vec(ins_data["alt"])
ins_vn   = vec(ins_data["vn"])
ins_ve   = vec(ins_data["ve"])
ins_vd   = vec(ins_data["vd"])
ins_fn   = vec(ins_data["fn"])
ins_fe   = vec(ins_data["fe"])
ins_fd   = vec(ins_data["fd"])
ins_Cnb  = ins_data["Cnb"]

map_info = "Map"
map_map  = map_data["map"]
map_xx   = deg2rad.(vec(map_data["xx"]))
map_yy   = deg2rad.(vec(map_data["yy"]))
map_alt  = map_data["alt"]
map_mask = MagNav.map_params(map_map,map_xx,map_yy)[2]

dt       = params["dt"]
baro_tau = params["baro_tau"]
acc_tau  = params["acc_tau"]
gyro_tau = params["gyro_tau"]
fogm_tau = params["meas_tau"]

tt       = vec(traj_data["tt"])
lat      = deg2rad.(vec(traj_data["lat"]))
lon      = deg2rad.(vec(traj_data["lon"]))
alt      = vec(traj_data["alt"])
vn       = vec(traj_data["vn"])
ve       = vec(traj_data["ve"])
vd       = vec(traj_data["vd"])
fn       = vec(traj_data["fn"])
fe       = vec(traj_data["fe"])
fd       = vec(traj_data["fd"])
Cnb      = traj_data["Cnb"]
mag_1_c  = vec(traj_data["mag_1_c"])
N        = length(lat)

traj = MagNav.Traj(N,dt,tt,lat,lon,alt,vn,ve,vd,fn,fe,fd,Cnb)
ins  = MagNav.INS( N,dt,tt,ins_lat,ins_lon,ins_alt,ins_vn,ins_ve,ins_vd,
                   ins_fn,ins_fe,ins_fd,ins_Cnb,zeros(3,3,N))

mapS = MagNav.MapS(map_info,map_map,map_xx,map_yy,map_alt,map_mask)
map_cache = Map_Cache(maps=[mapS])
(itp_mapS,der_mapS) = map_interpolate(mapS,:linear;return_vert_deriv=true) # linear to match MATLAB

# factor graph optimization (batch MAP smoother)
fgo_res = fgo(ins_lat,ins_lon,ins_alt,ins_vn,ins_ve,ins_vd,
              ins_fn,ins_fe,ins_fd,ins_Cnb,mag_1_c,dt,itp_mapS;
              P0       = P0,
              Qd       = Qd,
              R        = R,
              baro_tau = baro_tau,
              acc_tau  = acc_tau,
              gyro_tau = gyro_tau,
              fogm_tau = fogm_tau,
              core     = false)

# EKF for comparison (same model)
ekf_res = ekf(ins_lat,ins_lon,ins_alt,ins_vn,ins_ve,ins_vd,
              ins_fn,ins_fe,ins_fd,ins_Cnb,mag_1_c,dt,itp_mapS;
              P0       = P0,
              Qd       = Qd,
              R        = R,
              baro_tau = baro_tau,
              acc_tau  = acc_tau,
              gyro_tau = gyro_tau,
              fogm_tau = fogm_tau,
              core     = false)

# horizontal position DRMS [m] helper (INS position + estimated error vs truth)
drms(filt_res) = sqrt(mean(
    dlat2dn.(ins_lat .+ filt_res.x[1,:] .- lat, lat).^2 .+
    dlon2de.(ins_lon .+ filt_res.x[2,:] .- lon, lat).^2))
ins_drms = sqrt(mean(dlat2dn.(ins_lat .- lat, lat).^2 .+
                     dlon2de.(ins_lon .- lon, lat).^2))

@testset "fgo output structure" begin
    @test fgo_res isa FILTres
    @test size(fgo_res.x) == size(ekf_res.x)
    @test size(fgo_res.P) == size(ekf_res.P)
    @test size(fgo_res.r) == size(ekf_res.r)
    @test all(isfinite, fgo_res.x)
    @test all(isfinite, fgo_res.P)
    @test all(isfinite, fgo_res.r)
    @test all(diag(fgo_res.P[:,:,1])   .>= 0) # covariance diagonals non-negative
    @test all(diag(fgo_res.P[:,:,end]) .>= 0)
    @test fgo_res.P[:,:,1]   ≈ fgo_res.P[:,:,1]'   atol=1e-8 # symmetric
    @test fgo_res.P[:,:,end] ≈ fgo_res.P[:,:,end]' atol=1e-8
end

@testset "fgo navigation accuracy" begin
    # smoother must improve horizontal position over raw INS
    @test drms(fgo_res) < ins_drms
    # batch smoother should be no worse than the causal EKF (within a small margin)
    @test drms(fgo_res) <= 1.20 * drms(ekf_res)
    # smoother covariance should be no larger than filter covariance at t = 1
    @test fgo_res.P[1,1,1] <= ekf_res.P[1,1,1] * (1 + 1e-6)
end

@testset "fgo interfaces & options" begin
    @test fgo(ins,mag_1_c,itp_mapS)                            isa FILTres
    @test fgo(ins,mag_1_c,map_cache;core=true)                 isa FILTres
    @test fgo(ins,mag_1_c,itp_mapS;R=(1,10))                   isa FILTres
    @test fgo(ins,mag_1_c,itp_mapS;der_mapS,map_alt)           isa FILTres
    @test fgo(ins,mag_1_c,itp_mapS;n_iter=1)                   isa FILTres
    @test fgo(ins,mag_1_c,itp_mapS;n_iter=8,silent=false)      isa FILTres
    @test run_filt(traj,ins,mag_1_c,itp_mapS,:fgo;run_crlb=false) isa MagNav.FILTout
end

# sparse Gauss-Newton/QR (square-root information form) solver
gn_res = fgo(ins_lat,ins_lon,ins_alt,ins_vn,ins_ve,ins_vd,
             ins_fn,ins_fe,ins_fd,ins_Cnb,mag_1_c,dt,itp_mapS;
             P0       = P0,
             Qd       = Qd,
             R        = R,
             baro_tau = baro_tau,
             acc_tau  = acc_tau,
             gyro_tau = gyro_tau,
             fogm_tau = fogm_tau,
             core     = false,
             solver   = :gn)

@testset "fgo gn solver" begin
    @test gn_res isa FILTres
    @test all(isfinite, gn_res.x)
    # both solvers compute the same MAP estimate: horizontal position states
    # should agree closely (q_floor & conditioning allow small differences)
    n_err_rts = dlat2dn.(fgo_res.x[1,:],lat)
    n_err_gn  = dlat2dn.( gn_res.x[1,:],lat)
    e_err_rts = dlon2de.(fgo_res.x[2,:],lat)
    e_err_gn  = dlon2de.( gn_res.x[2,:],lat)
    @test sqrt(mean(abs2, n_err_gn .- n_err_rts)) < 1.0 # [m]
    @test sqrt(mean(abs2, e_err_gn .- e_err_rts)) < 1.0 # [m]
    # navigation accuracy comparable to RTS solution
    @test drms(gn_res) <= 1.20 * drms(fgo_res)
end

@testset "fgo robust kernels" begin
    @test fgo(ins,mag_1_c,itp_mapS;robust=:huber)             isa FILTres
    @test fgo(ins,mag_1_c,itp_mapS;robust=:cauchy)            isa FILTres
    @test fgo(ins,mag_1_c,itp_mapS;solver=:gn,robust=:huber)  isa FILTres
    @test MagNav.robust_weight(0.5,:huber ,1.345) ≈ 1.0
    @test MagNav.robust_weight(2.69,:huber,1.345) ≈ 0.5
    @test MagNav.robust_weight(2.385,:cauchy,2.385) ≈ 0.5
    @test MagNav.robust_weight(9.9,:none ,1.345) ≈ 1.0
    # a corrupted (outlier) segment should hurt the robust solution less
    mag_bad = copy(mag_1_c)
    mag_bad[41:50] .+= 500 # [nT] outlier segment
    res_l2 = fgo(ins,mag_bad,itp_mapS;
                 P0=P0,Qd=Qd,R=R,baro_tau=baro_tau,acc_tau=acc_tau,
                 gyro_tau=gyro_tau,fogm_tau=fogm_tau)
    res_hu = fgo(ins,mag_bad,itp_mapS;
                 P0=P0,Qd=Qd,R=R,baro_tau=baro_tau,acc_tau=acc_tau,
                 gyro_tau=gyro_tau,fogm_tau=fogm_tau,robust=:huber,n_iter=8)
    @test drms(res_hu) <= drms(res_l2)
end

# batch Tolles-Lawson estimation (compensation factors)
xyz0   = get_XYZ0(joinpath(@__DIR__,"test_data","test_data_traj.mat"),
                  :traj,:none;silent=true)
flux_a = xyz0.flux_a
(x0_TL,P0_TL,TL_sigma) = ekf_online_setup(flux_a,xyz0.mag_1_c;N_sigma=10)
(P0_o,Qd_o,R_o) = create_model(traj.dt,traj.lat[1];
                               vec_states = false,
                               TL_sigma   = TL_sigma,
                               P0_TL      = P0_TL)

@testset "fgo_online tests" begin
    res_o = fgo_online(ins,xyz0.mag_1_c,flux_a,itp_mapS,x0_TL,P0_o,Qd_o,R_o)
    @test res_o isa FILTres
    @test size(res_o.x,1) == 18 + length(x0_TL)
    @test all(isfinite, res_o.x)
    @test fgo_online(ins,xyz0.mag_1_c,flux_a,map_cache,x0_TL,P0_o,Qd_o,R_o;
                     robust=:huber) isa FILTres
    # fixed-lag sliding-window smoother (adaptive TL)
    res_w = fgo_online(ins,xyz0.mag_1_c,flux_a,itp_mapS,x0_TL,P0_o,Qd_o,R_o;
                       win=3.0,overlap=1.0)
    @test res_w isa FILTres
    @test size(res_w.x) == size(res_o.x)
    @test all(isfinite, res_w.x)
    @test run_filt(traj,ins,xyz0.mag_1_c,itp_mapS,:fgo_online;
                   P0=P0_o,Qd=Qd_o,R=R_o,flux=flux_a,x0_TL=x0_TL,
                   run_crlb=false) isa MagNav.FILTout
    # FGO-native observability gate (opt-in) runs and stays finite
    res_g = fgo_online(ins,xyz0.mag_1_c,flux_a,itp_mapS,x0_TL,P0_o,Qd_o,R_o;obs_gate=true)
    @test res_g isa FILTres
    @test all(isfinite, res_g.x)
    @test fgo_online(ins,xyz0.mag_1_c,flux_a,itp_mapS,x0_TL,P0_o,Qd_o,R_o;
                     win=3.0,overlap=1.0,obs_gate=true) isa FILTres
    # obs_collapse_index returns a value in [0,1]
    ρ = MagNav.obs_collapse_index(randn(18+length(x0_TL),50), length(x0_TL))
    @test 0 <= ρ <= 1
end

# sensor-error factors: inject a heading error into the scalar measurement
(_,_,psi_t) = dcm2euler(Cnb,:body2nav)
a1_true = 10.0
b1_true = -6.0
mag_head = mag_1_c .+ a1_true .* cos.(psi_t) .+ b1_true .* sin.(psi_t)

rms_r(fr) = sqrt(mean(abs2, fr.r))

@testset "fgo_sensor tests" begin
    fr_base = fgo_sensor(ins,mag_head,itp_mapS;P0=P0,Qd=Qd,R=R,
                         baro_tau=baro_tau,acc_tau=acc_tau,gyro_tau=gyro_tau,
                         fogm_tau=fogm_tau)
    fr_head = fgo_sensor(ins,mag_head,itp_mapS;P0=P0,Qd=Qd,R=R,
                         baro_tau=baro_tau,acc_tau=acc_tau,gyro_tau=gyro_tau,
                         fogm_tau=fogm_tau,n_harm=1)
    fr_both = fgo_sensor(ins,mag_head,itp_mapS;P0=P0,Qd=Qd,R=R,
                         baro_tau=baro_tau,acc_tau=acc_tau,gyro_tau=gyro_tau,
                         fogm_tau=fogm_tau,n_harm=2,cal_bias=true,robust=:huber)
    @test fr_base isa FILTres
    @test size(fr_base.x,1) == 18                # no sensor states
    @test size(fr_head.x,1) == 18 + 1            # 1 cosine heading coeff
    @test size(fr_both.x,1) == 18 + 2 + 3        # 2 cosine coeffs + 3 bias states
    @test all(isfinite, fr_head.x)
    @test all(isfinite, fr_both.x)
    # extra measurement DOF cannot increase the post-fit residual
    @test rms_r(fr_head) <= rms_r(fr_base) + 1e-6
    @test rms_r(fr_both) <= rms_r(fr_base) + 1e-6
    # heading coefficient state is estimated (non-trivial) & finite
    @test all(isfinite, fr_head.x[19,:])
    # full sensor-error model: heading {1,2,4} + bias + drift, dead-zone weighting
    fr_all = fgo_sensor(ins,mag_head,itp_mapS;P0=P0,Qd=Qd,R=R,
                        baro_tau=baro_tau,acc_tau=acc_tau,gyro_tau=gyro_tau,
                        fogm_tau=fogm_tau,n_harm=3,cal_bias=true,drift=true,
                        dead_zone=true)
    @test size(fr_all.x,1) == 18 + 3 + 3 + 1
    @test all(isfinite, fr_all.x)
    @test fgo_sensor(ins,mag_head,itp_mapS;n_harm=1,cal_bias=true)   isa FILTres
    @test fgo_sensor(ins,mag_head,itp_mapS;n_harm=1,heading=:yaw)    isa FILTres
    @test fgo_sensor(ins,mag_head,itp_mapS;drift=true,dead_zone=true) isa FILTres
end
