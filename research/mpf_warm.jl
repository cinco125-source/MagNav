##* Warm (not-cold-start) control for the fixed MPF recipe.
#
# mpf_fixed.jl runs the properly-tuned MPF+TL / MPF+TL+NN on the COLD START:
# uncompensated cabin Mag 4/5, so the filter must learn TL (and the NN) online
# from zero while it also localizes. This script is the mirror image: the SAME
# three methods (MPF+TL fixed, MPF+TL+NN fixed, FGO), same recipe (linear itp,
# log weights, thresh 0.1, roughening), same lines, but fed the COMPENSATED
# stinger mag_1_c -- the interference is already removed, so there is no cold
# start to survive. If MPF navigates here but diverges in mpf_fixed.jl, the
# failure is the cold-start transient, not the sampler.
#
# Usage: julia --project=. research/mpf_warm.jl

using MagNav
using CSV, DataFrames
using LinearAlgebra, Statistics
using Random: seed!
seed!(33)

include(joinpath(@__DIR__,"mpf_online.jl"))   # MPF+TL  (+ thresh/roughen)
include(joinpath(@__DIR__,"mpf_nn.jl"))        # MPF+TL+NN (+ thresh/roughen/init_ps)

# MEAS_VAR/FOGM_SIG/FOGM_TAU are const, defined by the mpf_nn.jl include above.
TERMS    = [:permanent,:induced,:eddy,:bias]
THR      = 0.1     # resample threshold (low = resample rarely)
RGH      = 5.0     # roughening [m]
INITPS   = 3.0     # PF-appropriate initial position sigma [m]
NP       = 1000

LINES = [(:Flt1003,1003.08), (:Flt1007,1007.06)]
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

function drms(traj, lat, lon; warm=600.0)
    m = (traj.tt .- traj.tt[1]) .>= warm
    sqrt(mean(dlat2dn.(lat[m] .- traj.lat[m], traj.lat[m]).^2 .+
              dlon2de.(lon[m] .- traj.lon[m], traj.lat[m]).^2))
end

results = DataFrame(flight=Symbol[],line=Float64[],mag=String[],method=String[],DRMS=Float64[])

for (fl,line) in LINES
    xyz  = get_XYZ(fl,df_flight;silent=true)
    ind  = get_ind(xyz,line,df_nav)
    mname= df_nav[(df_nav.flight.==fl).&(df_nav.line.==line),:map_name][1]
    mapS = get_map(mname,df_map)
    traj = get_traj(xyz,ind)
    ins  = get_ins(xyz,ind;N_zero_ll=1)
    itp_lin = map_interpolate(mapS, :linear)          # graceful out-of-grid for the PF
    flux = xyz.flux_d(ind)
    nTL  = size(create_TL_A(flux;terms=TERMS),2)
    (P0,Qd,R) = create_model(traj.dt,traj.lat[1];init_pos_sigma=INITPS,init_alt_sigma=1.0,
        init_vel_sigma=1.0,meas_var=MEAS_VAR,fogm_sigma=FOGM_SIG,fogm_tau=FOGM_TAU,
        vec_states=false,TL_sigma=fill(1.0,nTL),P0_TL=Matrix(Diagonal(fill(1.0,nTL))))

    println("\n$fl $line ($mname)")
    mag = xyz.mag_1_c[ind]                              # compensated stinger: no cold start
    tag = "Mag 1 comp"

    d_tl = mpf_online_drms(traj,ins,mag,flux,itp_lin,zeros(nTL),P0,Qd,R;
                terms=TERMS,num_part=NP,thresh=THR,roughen_m=RGH,core=true)
    d_nn = mpf_nn_drms(traj,ins,mag,flux,itp_lin;thresh=THR,roughen_m=RGH,init_ps=INITPS)
    d_fgo = try
        fr = fgo_online(ins,mag,flux,itp_lin,zeros(nTL),P0,Qd,R;terms=TERMS,core=true,
                        win=300.0,overlap=90.0,robust=:huber)
        drms(traj, (fo=MagNav.eval_filt(traj,ins,fr)).lat, fo.lon)
    catch e; @warn("fgo",e); Inf end

    push!(results,(fl,line,tag,"MPF+TL fixed",   round(d_tl, digits=1)))
    push!(results,(fl,line,tag,"MPF+TL+NN fixed",round(d_nn, digits=1)))
    push!(results,(fl,line,tag,"FGO",           round(d_fgo,digits=1)))
    println("  $tag  MPF+TL=$(round(d_tl,digits=1))  MPF+TL+NN=$(round(d_nn,digits=1))  FGO=$(round(d_fgo,digits=1)) m")
end

CSV.write(joinpath(@__DIR__,"mpf_warm_results.csv"), results)
println("\n=== fixed MPF recipe on the compensated (not-cold-start) stinger ===")
show(results, allrows=true, allcols=true); println()
println("wrote mpf_warm_results.csv")
