##* Dump baseline trajectories for the lat/lon comparison figure.
#
# The breadth study saved only DRMS scalars; the manuscript now wants the
# latitude/longitude error histories of every compared method on the
# representative line. Reruns the online-TL EKF and the EKF+TL+NN reference on
# line 1007.06 Mag 4 with the breadth-study settings and writes the
# trajectories to research/baseline_tracks_1007_06_m4.csv (radians).
#
# Usage: julia --project=. research/dump_baselines.jl

using MagNav
using CSV, DataFrames
using LinearAlgebra, Statistics
using Random: seed!
seed!(33)

include(joinpath(@__DIR__,"ekf_tlnn.jl"))

TERMS    = [:permanent,:induced,:eddy,:bias]
MEAS_VAR = 12.0^2
FOGM_SIG = 3.0
FOGM_TAU = 180.0

df_dir    = joinpath(@__DIR__,"..","examples","dataframes")
df_flight = DataFrame(CSV.File(joinpath(df_dir,"df_flight.csv")))
df_flight[!,:flight]   = Symbol.(df_flight[!,:flight])
df_flight[!,:xyz_type] = Symbol.(df_flight[!,:xyz_type])
df_flight[!,:xyz_file] = String.(df_flight[!,:xyz_file])
df_nav = DataFrame(CSV.File(joinpath(df_dir,"df_nav.csv")))
df_nav[!,:flight]   = Symbol.(df_nav[!,:flight])
df_nav[!,:map_name] = Symbol.(df_nav[!,:map_name])
df_map = DataFrame(CSV.File(joinpath(df_dir,"df_map.csv")))
df_map[!,:map_name] = Symbol.(df_map[!,:map_name])
df_map[!,:map_file] = String.(df_map[!,:map_file])

println("loading Flt1007..."); flush(stdout)
xyz  = get_XYZ(:Flt1007,df_flight;silent=true)
ind  = get_ind(xyz,1007.06,df_nav)
mapS = get_map(:Renfrew_395,df_map)
traj = get_traj(xyz,ind)
ins  = get_ins(xyz,ind;N_zero_ll=1)
(_,itp) = get_map_val(mapS,traj;return_itp=true)
flux = xyz.flux_d(ind)
mag  = xyz.mag_4_uc[ind]
nTL  = size(create_TL_A(flux;terms=TERMS),2)

(P0,Qd,R) = create_model(traj.dt,traj.lat[1];
    init_pos_sigma=0.1,init_alt_sigma=1.0,init_vel_sigma=1.0,
    meas_var=MEAS_VAR,fogm_sigma=FOGM_SIG,fogm_tau=FOGM_TAU,
    vec_states=false,TL_sigma=fill(1.0,nTL),
    P0_TL=Matrix(Diagonal(fill(1.0,nTL))))

println("running online-TL EKF..."); flush(stdout)
fo_ekf = run_filt(traj,ins,mag,itp,:ekf_online;P0=P0,Qd=Qd,R=R,
                  flux=flux,x0_TL=zeros(nTL),terms=TERMS,core=true,
                  run_crlb=false)

println("running EKF+TL+NN..."); flush(stdout)
# mirror ekf_tlnn_drms internals to keep the trajectory instead of the scalar
x  = create_TL_A(flux; terms=TLNN_TERMS)
(_,_,x_norm) = norm_sets(x); x_norm = Float32.(x_norm)
Mw   = min(traj.N, round(Int, 300/traj.dt))
xz   = zeros(18, Mw)
pred = MagNav.get_h(itp, xz, ins.lat[1:Mw], ins.lon[1:Mw], ins.alt[1:Mw]; core=true)
intf = mag[1:Mw] .- pred
y_norms = (Float32(median(intf)), Float32(std(intf)))
m     = MagNav.get_nn_m(size(x,2),1; hidden=TLNN_HIDDEN)
nx_nn = length(MagNav.destructure(m)[1])
P0_nn = Matrix(Diagonal(fill(TLNN_P0_SIGMA^2, nx_nn)))
nnsig = fill(TLNN_WEIGHT_Q, nx_nn)
(P0n,Qdn,Rn) = create_model(traj.dt,traj.lat[1];
    init_pos_sigma=0.1,init_alt_sigma=1.0,init_vel_sigma=1.0,
    meas_var=TLNN_MEAS_VAR,fogm_sigma=TLNN_FOGM_SIG,fogm_tau=TLNN_FOGM_TAU,
    vec_states=false,TL_sigma=nnsig,P0_TL=P0_nn)
frn   = ekf_online_nn(ins,mag,itp,x_norm,m,y_norms,P0n,Qdn,Rn;
                      fogm_tau=TLNN_FOGM_TAU,core=true)
fo_nn = MagNav.eval_filt(traj,ins,frn)

out = DataFrame(t=collect(0:traj.N-1).*traj.dt,
                true_lat=traj.lat, true_lon=traj.lon,
                ins_lat=ins.lat, ins_lon=ins.lon,
                ekf_lat=fo_ekf.lat, ekf_lon=fo_ekf.lon,
                nn_lat=fo_nn.lat, nn_lon=fo_nn.lon)
CSV.write(joinpath(@__DIR__,"baseline_tracks_1007_06_m4.csv"),out)
println("wrote baseline_tracks_1007_06_m4.csv")
