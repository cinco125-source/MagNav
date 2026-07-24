##* Window-length sweep on line 1007.06 (cold-start cabin mags).
# Uses the default marginalized (forward-filtered) window handoff.
# Fills in the gap between the short window and the static batch so the
# window-length-vs-accuracy curve is dense enough to be convincing: L_w in
# {2, 5, 10, 20, 40} min plus the static (full-line) batch. Same cold-start
# online-TL recipe as fgo_breadth.jl; DRMS after a 10-min warm-up.
#
# Usage: julia --project=. research/fgo_winlen.jl

using MagNav
using CSV, DataFrames
using LinearAlgebra, Statistics
using Random: seed!
seed!(33)

MEAS_VAR = 12.0^2
FOGM_SIG = 3.0
FOGM_TAU = 180.0
TERMS    = [:permanent,:induced,:eddy,:bias]
WINS_MIN = [2.0, 5.0, 10.0, 20.0, 40.0]   # sliding-window lengths; + static below

fl = :Flt1007; line = 1007.06

df_dir    = joinpath(@__DIR__,"..","examples","dataframes")
df_flight = DataFrame(CSV.File(joinpath(df_dir,"df_flight.csv")))
df_flight[!,:flight]   = Symbol.(df_flight[!,:flight])
df_flight[!,:xyz_type] = Symbol.(df_flight[!,:xyz_type])
df_flight[!,:xyz_file] = String.(df_flight[!,:xyz_file])
for (i,f) in enumerate(df_flight.flight)
    f == fl || continue
    df_flight.xyz_file[i] = MagNav.sgl_2020_train(f)
end
df_map = DataFrame(CSV.File(joinpath(df_dir,"df_map.csv")))
df_map[!,:map_name] = Symbol.(df_map[!,:map_name])
df_map[!,:map_file] = String.(df_map[!,:map_file])
for (i,m) in enumerate(df_map.map_name)
    df_map.map_file[i] = MagNav.ottawa_area_maps(m)
end
df_nav = DataFrame(CSV.File(joinpath(df_dir,"df_nav.csv")))
df_nav[!,:flight]   = Symbol.(df_nav[!,:flight])
df_nav[!,:map_name] = Symbol.(df_nav[!,:map_name])

xyz   = get_XYZ(fl,df_flight;silent=true)
ind   = get_ind(xyz,line,df_nav)
mname = df_nav[(df_nav.flight.==fl).&(df_nav.line.==line),:map_name][1]
mapS  = get_map(mname,df_map)
traj  = get_traj(xyz,ind)
ins   = get_ins(xyz,ind;N_zero_ll=1)
(_,itp) = get_map_val(mapS,traj;return_itp=true)
flux  = xyz.flux_d(ind)
nTL   = size(create_TL_A(flux;terms=TERMS),2)
line_min = traj.N*traj.dt/60
(P0,Qd,R) = create_model(traj.dt,traj.lat[1];init_pos_sigma=0.1,init_alt_sigma=1.0,
    init_vel_sigma=1.0,meas_var=MEAS_VAR,fogm_sigma=FOGM_SIG,fogm_tau=FOGM_TAU,
    vec_states=false,TL_sigma=fill(1.0,nTL),P0_TL=Matrix(Diagonal(fill(1.0,nTL))))

function drms(traj, lat, lon; warm=600.0)
    m  = (traj.tt .- traj.tt[1]) .>= warm
    dn = dlat2dn.(lat[m] .- traj.lat[m], traj.lat[m])
    de = dlon2de.(lon[m] .- traj.lon[m], traj.lat[m])
    sqrt(mean(dn.^2 .+ de.^2))
end

println("line 1007.06  ($mname, ~$(round(line_min,digits=0)) min)  cold-start window sweep")
results = DataFrame(win_min=Float64[], mag=String[], drms=Float64[])

for magsym in (:mag_4_uc,:mag_5_uc)
    mag = getfield(xyz,magsym)[ind]
    tag = replace(String(magsym),"mag_"=>"Mag ","_uc"=>"")
    for W in WINS_MIN
        d = NaN
        try
            fr = fgo_online(ins,mag,flux,itp,zeros(nTL),P0,Qd,R;terms=TERMS,
                            core=true,win=W*60.0,overlap=90.0,robust=:huber)
            fo = MagNav.eval_filt(traj,ins,fr)
            d  = drms(traj,fo.lat,fo.lon)
        catch e; @warn("win=$W min failed for $tag",e) end
        push!(results,(W,tag,round(d,digits=1)))
        println("  $tag  L_w=$(W) min  DRMS=$(round(d,digits=1)) m")
    end
    # static: one batch over the whole line (window >= line length, no overlap)
    d = NaN
    try
        fr = fgo_online(ins,mag,flux,itp,zeros(nTL),P0,Qd,R;terms=TERMS,
                        core=true,win=line_min*60+100,overlap=0.0,robust=:huber)
        fo = MagNav.eval_filt(traj,ins,fr)
        d  = drms(traj,fo.lat,fo.lon)
    catch e; @warn("static failed for $tag",e) end
    push!(results,(round(line_min,digits=0),tag,round(d,digits=1)))
    println("  $tag  static (whole line, $(round(line_min,digits=0)) min)  DRMS=$(round(d,digits=1)) m")
end

println("\n=== window-length sweep, line 1007.06 ===")
show(results;allrows=true,allcols=true); println()
CSV.write(joinpath(@__DIR__,"fgo_winlen_results.csv"),results)
println("\nwrote fgo_winlen_results.csv")
