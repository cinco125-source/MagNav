##* Minimal reproduction of the mpf_online failure seen in mpf_prior.jl.
# One case, no try/catch around the call, full traceback to stdout.
using MagNav, CSV, DataFrames, LinearAlgebra, Statistics
using Random: seed!

include(joinpath(@__DIR__,"mpf_online.jl"))

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

println("loading flight..."); flush(stdout)
xyz  = get_XYZ(:Flt1003,df_flight;silent=true)
ind  = get_ind(xyz,1003.02,df_nav)
mapS = get_map(:Eastern_395,df_map)
traj = get_traj(xyz,ind)
ins  = get_ins(xyz,ind;N_zero_ll=1)
(_,itp) = get_map_val(mapS,traj;return_itp=true)
flux = xyz.flux_d(ind)
TERMS = [:permanent,:induced,:eddy,:bias]
nTL  = size(create_TL_A(flux;terms=TERMS),2)
(P0,Qd,R) = create_model(traj.dt,traj.lat[1];
    init_pos_sigma=0.1,init_alt_sigma=1.0,init_vel_sigma=1.0,
    meas_var=144.0,fogm_sigma=3.0,fogm_tau=180.0,vec_states=false,
    TL_sigma=fill(1.0,nTL),P0_TL=Matrix(Diagonal(fill(100.0^2,nTL))))
println("nTL=$nTL nx=$(size(P0,1)); running mpf_online..."); flush(stdout)
seed!(33)
fr = mpf_online(ins,xyz.mag_4_uc[ind],flux,itp,zeros(nTL),P0,Qd,R;
                terms=TERMS,num_part=1000,core=true)
println("converge=", fr.converge)
