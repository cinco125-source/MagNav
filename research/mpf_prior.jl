##* MPF + online TL at a compensation prior scaled to the installation.
#
# The breadth study ran the RBPF baseline (research/mpf_online.jl) with the same
# compensation prior as every other estimator, sigma_beta = 1 per coefficient.
# The incremental FGO diagnosis showed that prior is 500-700 standard deviations
# too tight for an uncompensated cabin magnetometer, and the RBPF carries beta in
# its conditionally-linear Kalman block under exactly that prior. This rerun asks
# whether the particle filter's cold-start instability was the estimator or the
# prior: sigma_beta in {1, 100}, four counted lines, both cabin magnetometers,
# three seeds each, everything else at the breadth-study settings.
#
# Outputs research/mpf_prior_results.csv.
# Usage: julia --project=. research/mpf_prior.jl

using MagNav
using CSV, DataFrames
using LinearAlgebra, Statistics
using Random: seed!

include(joinpath(@__DIR__,"mpf_online.jl"))

TERMS    = [:permanent,:induced,:eddy,:bias]
MEAS_VAR = 12.0^2
FOGM_SIG = 3.0
FOGM_TAU = 180.0
NP       = 1000
SEEDS    = [33]
SIGMAS   = [1.0, 100.0]
# line selection via ARGS so two processes can split the flights
ALL      = [(:Flt1003,1003.02), (:Flt1003,1003.08),
            (:Flt1007,1007.02), (:Flt1007,1007.06)]
LINES    = isempty(ARGS) ? ALL : filter(x->String(x[1])==ARGS[1], ALL)

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

drms(traj,lat,lon;warm=600.0) = begin
    m0 = round(Int, warm/traj.dt) + 1
    m0 >= traj.N && (m0 = 1)
    dn = (lat[m0:end].-traj.lat[m0:end]).*MagNav.dlat2dn.(1.0,traj.lat[m0:end])
    de = (lon[m0:end].-traj.lon[m0:end]).*MagNav.dlon2de.(1.0,traj.lat[m0:end])
    sqrt(mean(dn.^2 .+ de.^2))
end

results = DataFrame(flight=Symbol[],line=Float64[],mag=String[],
                    sigma_beta=Float64[],seed=Int[],MPF_TL=Float64[],t_s=Float64[])

xyzs = Dict{Symbol,Any}()
for (fl,line) in LINES
    haskey(xyzs,fl) || (xyzs[fl] = get_XYZ(fl,df_flight;silent=true))
    xyz  = xyzs[fl]
    ind  = get_ind(xyz,line,df_nav)
    mname = df_nav[(df_nav.flight.==fl).&(df_nav.line.==line),:map_name][1]
    mapS = get_map(mname,df_map)
    traj = get_traj(xyz,ind)
    ins  = get_ins(xyz,ind;N_zero_ll=1)
    (_,itp) = get_map_val(mapS,traj;return_itp=true)
    flux = xyz.flux_d(ind)
    nTL  = size(create_TL_A(flux;terms=TERMS),2)

    for magsym in (:mag_4_uc,:mag_5_uc)
        mag = getfield(xyz,magsym)[ind]
        tag = replace(String(magsym),"mag_"=>"Mag ","_uc"=>"")
        for sb in SIGMAS
            (P0,Qd,R) = create_model(traj.dt,traj.lat[1];
                init_pos_sigma=0.1,init_alt_sigma=1.0,init_vel_sigma=1.0,
                meas_var=MEAS_VAR,fogm_sigma=FOGM_SIG,fogm_tau=FOGM_TAU,
                vec_states=false,TL_sigma=fill(1.0,nTL),
                P0_TL=Matrix(Diagonal(fill(sb^2,nTL))))
            for sd in (line==1007.06 ? [33,34,35] : SEEDS)
                seed!(sd)
                v = NaN; t = NaN
                try
                    t = @elapsed begin
                        fr = mpf_online(ins,mag,flux,itp,zeros(nTL),P0,Qd,R;
                                        terms=TERMS,num_part=NP,core=true)
                        fo = MagNav.eval_filt(traj,ins,fr)
                        v  = fr.c ? drms(traj,fo.lat,fo.lon) : Inf
                    end
                catch e
                    @warn("mpf_online failed",fl,line,tag,sb,sd,e)
                    v = Inf
                end
                push!(results,(fl,line,tag,sb,sd,round(v,digits=1),round(t,digits=1)))
                println("$fl $line $tag sigma=$sb seed=$sd  ->  $(round(v,digits=1)) m  ($(round(t,digits=1)) s)")
                flush(stdout)
            end
        end
    end
end

CSV.write(joinpath(@__DIR__,"mpf_prior_results_"*(isempty(ARGS) ? "all" : ARGS[1])*".csv"),results)
println("\nwrote mpf_prior_results.csv")
