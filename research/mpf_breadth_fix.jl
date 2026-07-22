##* Re-run the cold-start MPF+TL breadth with a PROPER config.
#
# The paper breadth table reports MPF+TL as "div." on all 8 cold-start cases, but that
# used thresh=0.1 and R=12 nT -- the over-confident R + rare resampling we showed
# collapse the PF. With the best config (standard resampling thresh=0.5, roughening 5,
# R matched ~40-100 nT) the MPF+TL stays BOUNDED (Mag 5: ~39-61 m; mpf_cold_R). This
# re-runs ALL 8 counted cases (4 lines x Mag 4/5) at that config, sweeping R in {40,100}
# so each case gets its best shot, with FGO (R=12) as reference. Goal: replace the
# "div." claim with honest bounded numbers.
#
# Outputs research/mpf_breadth_fix_results.csv. Usage: julia --project=. research/mpf_breadth_fix.jl

using MagNav
using CSV, DataFrames
using LinearAlgebra, Statistics
using Random: seed!
seed!(33)

include(joinpath(@__DIR__,"mpf_online.jl"))   # mpf_online_drms (+ thresh/roughen)

FOGM_SIG = 3.0
FOGM_TAU = 180.0
TERMS    = [:permanent,:induced,:eddy,:bias]
THR      = 0.5
RGH      = 5.0
NP       = 1000
MEAS_STDS = [40.0, 100.0]     # R sweep [nT]; report the best per case
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
                    meas_std_nT=Float64[],MPF_TL=Float64[],MPF_TL_best=Float64[],FGO=Float64[])

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

        (P0f,Qdf,Rf) = create_model(traj.dt,traj.lat[1];init_pos_sigma=3.0,init_alt_sigma=1.0,
            init_vel_sigma=1.0,meas_var=12.0^2,fogm_sigma=FOGM_SIG,fogm_tau=FOGM_TAU,
            vec_states=false,TL_sigma=fill(1.0,nTL),P0_TL=Matrix(Diagonal(fill(1.0,nTL))))
        d_fgo = try
            fr = fgo_online(ins,mag,flux,itp_lin,zeros(nTL),P0f,Qdf,Rf;terms=TERMS,core=true,
                            win=300.0,overlap=90.0,robust=:huber)
            drms(traj, (fo=MagNav.eval_filt(traj,ins,fr)).lat, fo.lon)
        catch e; @warn("fgo",e); Inf end

        best = Inf
        println("\n$fl $line ($mname) $tag  FGO=$(round(d_fgo,digits=1)) m")
        rows = Tuple[]
        for ms in MEAS_STDS
            (P0,Qd,R) = create_model(traj.dt,traj.lat[1];init_pos_sigma=3.0,init_alt_sigma=1.0,
                init_vel_sigma=1.0,meas_var=ms^2,fogm_sigma=FOGM_SIG,fogm_tau=FOGM_TAU,
                vec_states=false,TL_sigma=fill(1.0,nTL),P0_TL=Matrix(Diagonal(fill(1.0,nTL))))
            d_tl = mpf_online_drms(traj,ins,mag,flux,itp_lin,zeros(nTL),P0,Qd,R;
                        terms=TERMS,num_part=NP,thresh=THR,roughen_m=RGH,core=true)
            best = min(best, d_tl)
            push!(rows,(fl,line,tag,ms,round(d_tl,digits=1)))
            println("  R=$(ms) nT   MPF+TL=$(round(d_tl,digits=1)) m")
        end
        for r in rows
            push!(results,(r...,round(best,digits=1),round(d_fgo,digits=1)))
        end
        println("  -> MPF+TL best=$(round(best,digits=1)) m   FGO=$(round(d_fgo,digits=1)) m")
    end
end

CSV.write(joinpath(@__DIR__,"mpf_breadth_fix_results.csv"), results)
println("\n=== cold-start MPF+TL breadth, best config + R sweep ===")
show(results, allrows=true, allcols=true); println()
println("wrote mpf_breadth_fix_results.csv")
