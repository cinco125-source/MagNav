##* Seed statistics for the EKF+TL+NN baseline (review round 4, D6-adjacent).
#
# The NN-augmented EKF is the only randomly initialized baseline; the paper
# quotes single-seed numbers (48.8/18.3 m on 1007.06) and calls it
# seed-sensitive in the fig:latlon caption. This runs 10 seeds on 1007.06,
# both cabin magnetometers, and reports median and IQR so the text can carry a
# spread instead of an anecdote.
#
# Usage: julia --project=. research/nn_seed_stats.jl
# Outputs research/nn_seed_stats.csv

using MagNav
using CSV, DataFrames
using LinearAlgebra, Statistics
using Random: seed!

include(joinpath(@__DIR__,"ekf_tlnn.jl"))

FL, LINE = :Flt1007, 1007.06
SEEDS = 31:40

df_dir    = joinpath(@__DIR__,"..","examples","dataframes")
df_flight = DataFrame(CSV.File(joinpath(df_dir,"df_flight.csv")))
df_flight[!,:flight]   = Symbol.(df_flight[!,:flight])
df_flight[!,:xyz_type] = Symbol.(df_flight[!,:xyz_type])
df_flight[!,:xyz_file] = String.(df_flight[!,:xyz_file])
for (i,fl) in enumerate(df_flight.flight)
    fl == FL || continue
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

xyz  = get_XYZ(FL,df_flight;silent=true)
ind  = get_ind(xyz,LINE,df_nav)
mname= df_nav[(df_nav.flight.==FL).&(df_nav.line.==LINE),:map_name][1]
mapS = get_map(mname,df_map)
traj = get_traj(xyz,ind)
ins  = get_ins(xyz,ind;N_zero_ll=1)
(_,itp) = get_map_val(mapS,traj;return_itp=true)
flux = xyz.flux_d(ind)

results = DataFrame(seed=Int[],mag=String[],drms_m=Float64[])
for magsym in (:mag_4_uc,:mag_5_uc)
    mag = getfield(xyz,magsym)[ind]
    tag = replace(String(magsym),"mag_"=>"Mag ","_uc"=>"")
    for sd in SEEDS
        seed!(sd)
        d = try
            ekf_tlnn_drms(traj,ins,mag,flux,itp)
        catch e; @warn("nn seed $sd",e); Inf end
        push!(results,(sd,tag,round(d,digits=1)))
        println("$tag seed=$sd  EKF+TL+NN=$(round(d,digits=1)) m"); flush(stdout)
        CSV.write(joinpath(@__DIR__,"nn_seed_stats.csv"),results)
    end
end
for tag in unique(results.mag)
    v = filter(isfinite, results[results.mag.==tag,:drms_m])
    println("$tag  median=$(round(median(v),digits=1))  ",
            "IQR=$(round(quantile(v,0.25),digits=1))-",
            "$(round(quantile(v,0.75),digits=1))  n=$(length(v))")
end
println("done")
