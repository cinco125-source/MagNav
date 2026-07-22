##* Fairness check: does the cold-start MPF+TL still diverge with a properly tuned R?
#
# The cold-start baseline (mpf_fixed, mpf_nn, fgo_breadth) ran the MPF+TL at R=12 nT
# (meas_var=12^2) -- the same over-confident R we later showed collapses the PF on the
# offline-NN study. The EKF is R-insensitive so R=12 is fine for it, but it is unfair
# to the MPF. Re-run the cold-start MPF+TL on the raw uncompensated Mag 5 with our BEST
# config (standard resampling thresh=0.5, roughening 5) and SWEEP R in {12,40,100,300}
# nT. If it still diverges at every R, the cold-start failure is not an R artifact:
# before the online TL converges the effective residual is huge and time-varying, so
# no fixed R matches. FGO (R=12) is the bounded reference.
#
# Outputs research/mpf_cold_R_results.csv. Usage: julia --project=. research/mpf_cold_R.jl

using MagNav
using CSV, DataFrames
using LinearAlgebra, Statistics
using Random: seed!
seed!(33)

include(joinpath(@__DIR__,"mpf_online.jl"))   # mpf_online_drms (+ thresh/roughen)

FOGM_SIG = 3.0
FOGM_TAU = 180.0
TERMS    = [:permanent,:induced,:eddy,:bias]
THR      = 0.5      # standard resampling (best from the sweep)
RGH      = 5.0      # roughening (necessary)
NP       = 1000
MEAS_STDS = [12.0, 40.0, 100.0, 300.0]   # R sweep [nT]
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

function drms(traj, lat, lon; warm=600.0, div_thresh=1e4)
    m = (traj.tt .- traj.tt[1]) .>= warm
    d = sqrt(mean(dlat2dn.(lat[m] .- traj.lat[m], traj.lat[m]).^2 .+
                  dlon2de.(lon[m] .- traj.lon[m], traj.lat[m]).^2))
    (isfinite(d) && d < div_thresh) ? d : Inf
end

results = DataFrame(flight=Symbol[],line=Float64[],mag=String[],
                    meas_std_nT=Float64[],MPF_TL=Float64[],FGO=Float64[])

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
    mag  = xyz.mag_5_uc[ind]     # raw uncompensated cabin mag = cold start

    # FGO reference at the paper R=12 (R-insensitive, robust)
    (P0f,Qdf,Rf) = create_model(traj.dt,traj.lat[1];init_pos_sigma=3.0,init_alt_sigma=1.0,
        init_vel_sigma=1.0,meas_var=12.0^2,fogm_sigma=FOGM_SIG,fogm_tau=FOGM_TAU,
        vec_states=false,TL_sigma=fill(1.0,nTL),P0_TL=Matrix(Diagonal(fill(1.0,nTL))))
    d_fgo = try
        fr = fgo_online(ins,mag,flux,itp_lin,zeros(nTL),P0f,Qdf,Rf;terms=TERMS,core=true,
                        win=300.0,overlap=90.0,robust=:huber)
        drms(traj, (fo=MagNav.eval_filt(traj,ins,fr)).lat, fo.lon)
    catch e; @warn("fgo",e); Inf end

    println("\n$fl $line ($mname)  FGO=$(round(d_fgo,digits=1)) m  (best-config MPF+TL, R sweep)")
    for ms in MEAS_STDS
        (P0,Qd,R) = create_model(traj.dt,traj.lat[1];init_pos_sigma=3.0,init_alt_sigma=1.0,
            init_vel_sigma=1.0,meas_var=ms^2,fogm_sigma=FOGM_SIG,fogm_tau=FOGM_TAU,
            vec_states=false,TL_sigma=fill(1.0,nTL),P0_TL=Matrix(Diagonal(fill(1.0,nTL))))
        d_tl = mpf_online_drms(traj,ins,mag,flux,itp_lin,zeros(nTL),P0,Qd,R;
                    terms=TERMS,num_part=NP,thresh=THR,roughen_m=RGH,core=true)
        push!(results,(fl,line,"Mag 5",ms,round(d_tl,digits=1),round(d_fgo,digits=1)))
        println("  R=$(ms) nT   MPF+TL=$(round(d_tl,digits=1)) m")
    end
end

CSV.write(joinpath(@__DIR__,"mpf_cold_R_results.csv"), results)
println("\n=== cold-start MPF+TL, best config + R sweep (raw Mag 5) ===")
show(results, allrows=true, allcols=true); println()
println("wrote mpf_cold_R_results.csv")
