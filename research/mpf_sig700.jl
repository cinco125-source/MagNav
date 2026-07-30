##* MPF+TL cold-start retest: sigma_beta alone, everything else unchanged.
#
# The MPF audit's D1: the breadth runs used P0_TL = I (sigma_beta = 1 nT)
# against true coefficients of 550-720, making the predictive variance V wrong
# by ~5 orders of magnitude and turning the weight update into a directed
# gradient walk on the map. This runs the single-change confirmation the audit
# asks for: P0_TL = 700^2 I, every other setting identical to
# mpf_breadth_fix.jl (thresh 0.5, roughen 5, NP 1000, R = 40 nT, seed 33,
# R_gain inflation still on). If 1007.06 Mag 4 drops from 1227 m to O(10^2) m,
# D1 is confirmed as the dominant term.
#
# Usage: julia --project=. research/mpf_sig700.jl [Flt1007|Flt1003]
# Outputs research/mpf_sig700_results_<flight>.csv

using MagNav
using CSV, DataFrames
using LinearAlgebra, Statistics
using Random: seed!
seed!(33)

include(joinpath(@__DIR__,"mpf_online.jl"))   # mpf_online_drms

FOGM_SIG = 3.0
FOGM_TAU = 180.0
TERMS    = [:permanent,:induced,:eddy,:bias]
THR      = 0.5
RGH      = 5.0
NP       = 1000
MEAS_STD = 40.0
SIG_BETA = 700.0
LINES = [(:Flt1007,1007.06), (:Flt1007,1007.02),
         (:Flt1003,1003.02), (:Flt1003,1003.08)]
if !isempty(ARGS)
    want = Symbol(ARGS[1])
    LINES = filter(x->first(x)==want, LINES)
end
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
for (i,mn) in enumerate(df_map.map_name)
    df_map.map_file[i] = MagNav.ottawa_area_maps(mn)
end
df_nav = DataFrame(CSV.File(joinpath(df_dir,"df_nav.csv")))
df_nav[!,:flight]   = Symbol.(df_nav[!,:flight])
df_nav[!,:map_name] = Symbol.(df_nav[!,:map_name])

function drms(traj, lat, lon; warm=600.0, div_thresh=1e4)
    m = (traj.tt .- traj.tt[1]) .>= warm
    d = sqrt(mean(dlat2dn.(lat[m] .- traj.lat[m], traj.lat[m]).^2 .+
                  dlon2de.(lon[m] .- traj.lon[m], traj.lat[m]).^2))
    (isfinite(d) && d < div_thresh) ? d : Inf
end

results = DataFrame(flight=Symbol[],line=Float64[],mag=String[],
                    sigma_beta=Float64[],meas_std_nT=Float64[],MPF_TL=Float64[])

for (fl,line) in LINES
    xyz  = get_XYZ(fl,df_flight;silent=true)
    ind  = get_ind(xyz,line,df_nav)
    mname= df_nav[(df_nav.flight.==fl).&(df_nav.line.==line),:map_name][1]
    mapS = get_map(mname,df_map)
    traj = get_traj(xyz,ind)
    ins  = get_ins(xyz,ind;N_zero_ll=1)
    itp_lin = map_interpolate(mapS, :linear)
    flux = xyz.flux_d(ind)
    nTL  = size(create_TL_A(flux;terms=TERMS),2)

    for magsym in (:mag_4_uc, :mag_5_uc)
        mag = getfield(xyz,magsym)[ind]
        tag = replace(String(magsym),"mag_"=>"Mag ","_uc"=>"")
        (P0,Qd,R) = create_model(traj.dt,traj.lat[1];init_pos_sigma=3.0,init_alt_sigma=1.0,
            init_vel_sigma=1.0,meas_var=MEAS_STD^2,fogm_sigma=FOGM_SIG,fogm_tau=FOGM_TAU,
            vec_states=false,TL_sigma=fill(1.0,nTL),
            P0_TL=Matrix(Diagonal(fill(SIG_BETA^2,nTL))))
        d_tl = try
            mpf_online_drms(traj,ins,mag,flux,itp_lin,zeros(nTL),P0,Qd,R;
                        terms=TERMS,num_part=NP,thresh=THR,roughen_m=RGH,core=true)
        catch e; @warn("mpf",e); Inf end
        push!(results,(fl,line,tag,SIG_BETA,MEAS_STD,round(d_tl,digits=1)))
        println("$fl $line $tag  sigma_beta=$SIG_BETA R=$MEAS_STD  ",
                "MPF+TL=$(round(d_tl,digits=1)) m"); flush(stdout)
        CSV.write(joinpath(@__DIR__,"mpf_sig700_results_$(flights[1]).csv"),results)
    end
end
println("done")
