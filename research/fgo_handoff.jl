##* C-2 (marginalized handoff): smoothed vs forward-filtered window handoff.
#
# The fixed-lag window commits its leading stride and carries a prior to the next
# window. The legacy handoff carries the SMOOTHED boundary state, which has already
# absorbed the trailing overlap measurements; the next window re-processes those
# overlap epochs, counting their information twice. The :filtered handoff instead
# carries the FORWARD-FILTERED marginal at the boundary, which has not seen the
# overlap — for a chain this is exactly the marginalization of the departing
# sub-graph (square-root SAM / Schur complement), so the overlap is counted once and
# the handoff is consistent.
#
# This compares the two handoffs on the breadth lines (cold start, uncompensated
# cabin Mag 4/5), win=300 s, overlap=90 s, Huber. If DRMS is essentially unchanged,
# the marginalized handoff removes the double-count at no accuracy cost — the paper
# can drop the "double-count" caveat and describe a consistent scheme.
#
# Usage: julia --project=. research/fgo_handoff.jl

using MagNav
using CSV, DataFrames
using LinearAlgebra, Statistics
using Random: seed!
seed!(33)

MEAS_VAR = 12.0^2
FOGM_SIG = 3.0
FOGM_TAU = 180.0
TERMS    = [:permanent,:induced,:eddy,:bias]
WIN      = 300.0
OVERLAP  = 90.0

LINES = [(:Flt1007,1007.06), (:Flt1007,1007.02),
         (:Flt1003,1003.02), (:Flt1003,1003.08)]
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
for (i,map_name) in enumerate(df_map.map_name)
    df_map.map_file[i] = MagNav.ottawa_area_maps(map_name)
end
df_nav = DataFrame(CSV.File(joinpath(df_dir,"df_nav.csv")))
df_nav[!,:flight]   = Symbol.(df_nav[!,:flight])
df_nav[!,:map_name] = Symbol.(df_nav[!,:map_name])

xyz_cache = Dict{Symbol,Any}()
getxyz(fl) = get!(()->get_XYZ(fl,df_flight;silent=true), xyz_cache, fl)

function drms(traj, lat, lon; warm=600.0)
    m  = (traj.tt .- traj.tt[1]) .>= warm
    dn = dlat2dn.(lat[m] .- traj.lat[m], traj.lat[m])
    de = dlon2de.(lon[m] .- traj.lon[m], traj.lat[m])
    sqrt(mean(dn.^2 .+ de.^2))
end

function fgo_drms(traj,ins,mag,flux,itp,nTL,P0,Qd,R;handoff=:smoothed)
    fr = fgo_online(ins,mag,flux,itp,zeros(nTL),P0,Qd,R;terms=TERMS,core=true,
                    win=WIN,overlap=OVERLAP,robust=:huber,handoff=handoff)
    fo = MagNav.eval_filt(traj,ins,fr)
    drms(traj,fo.lat,fo.lon)
end

results = DataFrame(flight=Symbol[],line=Float64[],mag=String[],
                    smoothed=Float64[],filtered=Float64[],delta=Float64[])

for (fl,line) in LINES
    xyz   = getxyz(fl)
    ind   = get_ind(xyz,line,df_nav)
    mname = df_nav[(df_nav.flight.==fl).&(df_nav.line.==line),:map_name][1]
    mapS  = get_map(mname,df_map)
    traj  = get_traj(xyz,ind)
    ins   = get_ins(xyz,ind;N_zero_ll=1)
    (_,itp) = get_map_val(mapS,traj;return_itp=true)
    flux  = xyz.flux_d(ind)
    nTL   = size(create_TL_A(flux;terms=TERMS),2)
    (P0,Qd,R) = create_model(traj.dt,traj.lat[1];init_pos_sigma=0.1,init_alt_sigma=1.0,
        init_vel_sigma=1.0,meas_var=MEAS_VAR,fogm_sigma=FOGM_SIG,fogm_tau=FOGM_TAU,
        vec_states=false,TL_sigma=fill(1.0,nTL),P0_TL=Matrix(Diagonal(fill(1.0,nTL))))

    println("\n$fl $line  ($mname, N=$(traj.N))")
    for magsym in (:mag_4_uc,:mag_5_uc)
        mag = getfield(xyz,magsym)[ind]
        tag = replace(String(magsym),"mag_"=>"Mag ","_uc"=>"")
        ds = NaN; df_ = NaN
        try; ds  = fgo_drms(traj,ins,mag,flux,itp,nTL,P0,Qd,R;handoff=:smoothed)
        catch e; @warn("smoothed handoff failed $fl $line $tag",e) end
        try; df_ = fgo_drms(traj,ins,mag,flux,itp,nTL,P0,Qd,R;handoff=:filtered)
        catch e; @warn("filtered handoff failed $fl $line $tag",e) end
        push!(results,(fl,line,tag,round(ds,digits=2),round(df_,digits=2),
                       round(df_-ds,digits=2)))
        println("  $tag  smoothed=$(round(ds,digits=1))  ",
                "filtered(marginalized)=$(round(df_,digits=1))  ",
                "Δ=$(round(df_-ds,digits=2)) m")
    end
end

println("\n=== C-2 handoff: smoothed vs forward-filtered (marginalized), win=$(Int(WIN))/ov=$(Int(OVERLAP)) ===")
show(results;allrows=true,allcols=true); println()
CSV.write(joinpath(@__DIR__,"fgo_handoff_results.csv"),results)
fin = filter(r->isfinite(r.smoothed) && isfinite(r.filtered), results)
if nrow(fin) > 0
    println("\nmax |Δ| = $(round(maximum(abs.(fin.delta)),digits=2)) m over $(nrow(fin)) cases; ",
            "mean Δ = $(round(mean(fin.delta),digits=2)) m (filtered − smoothed).")
end
