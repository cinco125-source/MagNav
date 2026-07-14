##* Factor Graph Optimization (FGO) example for airborne MagNav
# This script solves the same airborne magnetic anomaly navigation problem as
# `simple_magnav.jl`, but compares the causal Extended Kalman Filter (`:ekf`)
# with the batch Factor Graph Optimization / MAP smoother (`:fgo`).
#
# FGO poses the whole flight as a single optimization over the error states:
#     minimize   ‖x_1‖²_{P0^-1}                                 (prior factor)
#              + Σ_t ‖x_{t+1} - Φ_t x_t‖²_{Qd^-1}               (process factors)
#              + Σ_t ‖meas_t - h(x_t)‖²_{R^-1}                  (measurement factors)
# The Pinson error model is a linear-Gaussian chain, so this MAP estimate is
# obtained exactly with an iterated fixed-interval (RTS) smoother that reuses the
# EKF model. Because every estimate sees all measurements (past AND future), FGO
# typically achieves lower navigation error than the EKF, especially early on.
#
# Usage: start Julia in this folder and `include("fgo_example.jl")`.

##* Setup
cd(@__DIR__)
using Pkg
# Pkg.add("MagNav") # as needed
using MagNav
using Random: seed!
using Statistics: mean
dir = "simple_magnav_data";

##* Load example data (HDF5)
seed!(33); # for reproducibility
xyz  = get_XYZ0(joinpath(dir,"xyz.h5"));
mapS = get_map(joinpath(dir,"map.h5"));

##* Create MagNav filter model
(P0,Qd,R) = create_model(xyz.traj.dt, xyz.traj.lat[1];
                         init_pos_sigma = 3.0,
                         init_alt_sigma = 0.001,
                         init_vel_sigma = 0.01,
                         init_att_sigma = deg2rad(0.01),
                         meas_var       = 3.0^2,
                         fogm_sigma     = 3.0,
                         fogm_tau       = 600.0);

itp_mapS = map_interpolate(mapS); # map interpolation function

##* horizontal position error helper [m]
function drms(traj, ins, filt_out)
    dn = MagNav.dlat2dn.(filt_out.lat .- traj.lat, traj.lat)
    de = MagNav.dlon2de.(filt_out.lon .- traj.lon, traj.lat)
    sqrt(mean(dn.^2 .+ de.^2))
end

##* Run the causal EKF
(_,_,ekf_out) = run_filt(xyz.traj, xyz.ins, xyz.mag_1_c, itp_mapS, :ekf;
                         P0=P0, Qd=Qd, R=R, core=false);

##* Run the batch Factor Graph Optimization (MAP smoother)
(_,_,fgo_out) = run_filt(xyz.traj, xyz.ins, xyz.mag_1_c, itp_mapS, :fgo;
                         P0=P0, Qd=Qd, R=R, core=false, n_iter=5);

##* Compare navigation accuracy
println("horizontal position DRMS [m]")
println("  EKF (causal filter)      : ", round(drms(xyz.traj, xyz.ins, ekf_out); digits=3))
println("  FGO (batch MAP smoother) : ", round(drms(xyz.traj, xyz.ins, fgo_out); digits=3))

##* Plot both results
p_ekf = plot_filt(xyz.traj, xyz.ins, ekf_out; show_plot=false)[1]
p_fgo = plot_filt(xyz.traj, xyz.ins, fgo_out; show_plot=false)[1]
