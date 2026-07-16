##* Do the physical sensor-error factors help on REAL data? (curiosity test)
#
# The sensor-error factors (OPM heading harmonics, hard-iron, drift, dead-zone
# reweighting) are validated in simulation with model-matched injection. This
# script asks the natural follow-up: attach the same factor nodes on real SGL
# 2020 lines and see whether DRMS improves.
#
# Design notes (why these sensors):
#  - fgo_sensor carries NO Tolles-Lawson chain, so on the uncompensated cabin
#    mags (Mag 4/5) it would face the full platform field and fail like any
#    TL-less filter. The fair real-data test is on the SGL-COMPENSATED sensors,
#    where TL-able interference is already removed and what remains is map error
#    plus genuine residual sensor error: Mag 1 (stinger OPM, best case) and
#    Mag 3 (cabin OPM, compensated but with a larger non-TL remainder).
#  - Physics prediction, stated before running: gains should be small-to-null.
#    For a body-axis OPM, cos(psi) equals a field direction cosine, so the
#    n={1,2} heading harmonics and the hard-iron term lie (nearly) inside the
#    span TL compensation has already removed; the Ottawa field inclination
#    (~70 deg) keeps a horizontal optical axis far from the dead zone; and the
#    FOGM disturbance state already absorbs slow drift. A null result validates
#    the paper's scoping (sensor factors matter when sensor error dominates,
#    e.g. an uncalibrated OPM); a gain would be a new result.
#
# Outputs research/fgo_sensor_real_results.csv.
# Usage: julia --project=. research/fgo_sensor_real.jl

using MagNav
using CSV, DataFrames
using LinearAlgebra, Statistics
using Random: seed!
seed!(7)

MEAS_VAR = 1.0^2      # compensated sensors: paper-benchmark noise level
FOGM_SIG = 1.0
FOGM_TAU = 600.0
AXIS     = [1.0,0.0,0.0]   # OPM optical axis assumed body-x (as in the ablation)
LINES    = [(:Flt1003,1003.02), (:Flt1007,1007.06)]
SENSORS  = (:mag_1_c,:mag_3_c)
flights  = unique(first.(LINES))

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

function drms_ll(traj, lat, lon; warm=600.0)
    m  = (traj.tt .- traj.tt[1]) .>= warm
    dn = dlat2dn.(lat[m] .- traj.lat[m], traj.lat[m])
    de = dlon2de.(lon[m] .- traj.lon[m], traj.lat[m])
    sqrt(mean(dn.^2 .+ de.^2))
end

# (name, n_harm, cal_bias, drift, dead_zone); n_harm=3 -> orders {1,2,4}
CASES = [("plain fgo",        0,false,false,false),
         ("+heading {1,2,4}", 3,false,false,false),
         ("full sensor model",3,true, true, true )]

results = DataFrame(flight=Symbol[],line=Float64[],mag=String[],config=String[],
                    robust=String[],DRMS=Float64[],N=Int[],t=Float64[])

for (fl,line) in LINES
    xyz   = getxyz(fl)
    ind   = get_ind(xyz,line,df_nav)
    mname = df_nav[(df_nav.flight.==fl).&(df_nav.line.==line),:map_name][1]
    mapS  = get_map(mname,df_map)
    traj  = get_traj(xyz,ind)
    ins   = get_ins(xyz,ind;N_zero_ll=1)
    (_,itp) = get_map_val(mapS,traj;return_itp=true)
    (P0,Qd,R) = create_model(traj.dt,traj.lat[1];init_pos_sigma=0.1,
        init_alt_sigma=1.0,init_vel_sigma=1.0,meas_var=MEAS_VAR,
        fogm_sigma=FOGM_SIG,fogm_tau=FOGM_TAU)
    println("\n$fl $line  ($mname, ~$(round(traj.N*traj.dt/60,digits=0)) min)")
    for magsym in SENSORS
        mag = getfield(xyz,magsym)[ind]
        tag = replace(String(magsym),"mag_"=>"Mag ","_c"=>"")
        for rob in (:none,:huber), (name,nh,cb,dr,dz) in CASES
            # plain fgo has no sensor states; run it once per robust setting
            d = NaN; tel = NaN
            try
                tel = @elapsed begin
                    fr = fgo_sensor(ins,mag,itp;P0=P0,Qd=Qd,R=R,core=true,
                                    n_harm=nh,heading=:geometry,axis=AXIS,
                                    cal_bias=cb,drift=dr,dead_zone=dz,
                                    robust=rob,n_iter=6)
                    fo = MagNav.eval_filt(traj,ins,fr)
                    d  = drms_ll(traj,fo.lat,fo.lon)
                end
            catch e
                @warn("$fl $line $tag $name ($rob) failed",e)
            end
            push!(results,(fl,line,tag,name,String(rob),round(d,digits=1),
                           traj.N,round(tel,digits=2)))
            println("  $tag  $(rpad(name,20)) robust=$(rpad(String(rob),6))",
                    " DRMS=$(round(d,digits=1)) m  ($(round(tel,digits=1)) s)")
        end
    end
end

println("\n=== sensor-error factors on real (compensated) SGL sensors ===")
show(results;allrows=true,allcols=true); println()
CSV.write(joinpath(@__DIR__,"fgo_sensor_real_results.csv"),results)
println("wrote fgo_sensor_real_results.csv")
