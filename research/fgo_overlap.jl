##* C-2 overlap double-counting: does re-processing the overlap change DRMS?
#
# The fixed-lag window commits its leading `stride = win - overlap` samples and
# carries the SMOOTHED boundary state forward as the next window's prior. That
# boundary state already absorbed the trailing `overlap` measurements, which the
# next window then re-processes with fresh factors — the overlap measurements are
# counted twice (a mild information double-count). Proper fixed-lag smoothers
# marginalize the dropped states (Schur complement) instead.
#
# Question for the paper: does this double-count actually move the point estimate
# (DRMS), or only the covariance? We sweep the overlap from 0 (no double-count, no
# look-ahead) up through the paper's 90 s and bound its effect on DRMS. If DRMS is
# essentially flat in overlap, the "look-ahead helps, double-count is negligible on
# the mean" claim is empirically supported and only the covariance wording needs a
# caveat. If DRMS moves materially, a proper (marginalized / forward-filtered)
# handoff is warranted.
#
# Cold start, raw uncompensated cabin Mag 4 / Mag 5, primary free-flight line
# 1007.06 (plus survey line 1003.08 as a second block). DRMS after 10-min warm-up.
#
# Usage: julia --project=. research/fgo_overlap.jl

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
OVERLAPS = [0.0, 30.0, 60.0, 90.0, 120.0, 150.0]   # 90 = paper baseline

LINES = [(:Flt1007,1007.06), (:Flt1003,1003.08)]
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

results = DataFrame(flight=Symbol[],line=Float64[],mag=String[],
                    overlap=Float64[],stride=Float64[],DRMS=Float64[])

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

    println("\n$fl $line  ($mname, ~$(round(traj.N*traj.dt/60,digits=0)) min, N=$(traj.N))")
    for magsym in (:mag_4_uc,:mag_5_uc)
        mag = getfield(xyz,magsym)[ind]
        tag = replace(String(magsym),"mag_"=>"Mag ","_uc"=>"")
        print("  $tag: ")
        for ov in OVERLAPS
            d = NaN
            try
                fr = fgo_online(ins,mag,flux,itp,zeros(nTL),P0,Qd,R;terms=TERMS,
                                core=true,win=WIN,overlap=ov,robust=:huber)
                fo = MagNav.eval_filt(traj,ins,fr)
                d  = drms(traj,fo.lat,fo.lon)
            catch e; @warn("fgo win=$WIN overlap=$ov failed for $fl $line $tag",e) end
            push!(results,(fl,line,tag,ov,WIN-ov,round(d,digits=2)))
            print("ov=$(Int(ov))→$(round(d,digits=1))  ")
        end
        println()
    end
end

println("\n=== C-2: FGO DRMS [m] vs window overlap (win=$(Int(WIN)) s), cold-start cabin mags ===")
show(results;allrows=true,allcols=true); println()
CSV.write(joinpath(@__DIR__,"fgo_overlap_results.csv"),results)

# spread of DRMS across the overlap sweep per mag-line: how much does the
# double-count (overlap>0) move the point estimate vs the no-overlap handoff?
println("\nPer mag-line: DRMS at overlap=0 (no double-count) vs overlap=90 (paper) and full spread")
for g in groupby(results, [:flight,:line,:mag])
    fin = filter(isfinite, g.DRMS)
    isempty(fin) && continue
    d0  = g[g.overlap .== 0.0,   :DRMS]
    d90 = g[g.overlap .== 90.0,  :DRMS]
    lo, hi = minimum(fin), maximum(fin)
    d0v  = isempty(d0)  ? NaN : d0[1]
    d90v = isempty(d90) ? NaN : d90[1]
    println("  $(g.flight[1]) $(g.line[1]) $(g.mag[1]): ",
            "ov0=$(round(d0v,digits=1))  ov90=$(round(d90v,digits=1))  ",
            "range=[$(round(lo,digits=1)), $(round(hi,digits=1))]  ",
            "Δ(90−0)=$(round(d90v-d0v,digits=1)) m  ",
            "spread=$(round(hi-lo,digits=1)) m ($(round(100*(hi-lo)/lo,digits=1))%)")
end
