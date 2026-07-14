##* FGO-online vs the cold-start online EKF+TL+NN of Hager et al. (2026)
# The paper "Airborne MagNav with NN-Augmented Online Calibration" (Hager et al.,
# arXiv 2603.08265) augments an EKF with Tolles-Lawson (TL) coefficients AND a
# residual neural network (NN), learned ONLINE from a COLD START (no calibration
# flight, no NN pre-training), stabilized by a natural-gradient / residual-
# constraint design. Its exact architecture, feature sets, and covariance tuning
# are NOT publicly released.
#
# HONESTY NOTE — we do NOT re-run the paper's method here. A naive cold-start of
# an NN-in-EKF (untrained network, generic covariances) diverges — which is
# exactly the instability the paper's design exists to prevent — so re-running it
# untuned would be a strawman, not a reproduction. Instead we (1) run OUR batch
# FGO-online (TL coefficients as factor-graph nodes) from the same cold start on
# the same data, and (2) print the paper's PUBLISHED cold-start DRMS values as a
# cited reference line. The comparison is indicative, not a controlled
# reproduction of their filter.
#
# Data: SGL 2020, the paper's primary line 1007.06, full length, uncompensated
# cabin magnetometers, DRMS after a 10-min warm-up (paper convention).
#
# Usage: julia --project=. research/paper_baseline.jl

using MagNav
using CSV, DataFrames
using LinearAlgebra, Statistics
using Random: seed!
seed!(33)

df_dir    = joinpath(@__DIR__,"..","examples","dataframes")
df_flight = DataFrame(CSV.File(joinpath(df_dir,"df_flight.csv")))
df_flight[!,:flight]   = Symbol.(df_flight[!,:flight])
df_flight[!,:xyz_type] = Symbol.(df_flight[!,:xyz_type])
df_flight[!,:xyz_file] = String.(df_flight[!,:xyz_file])
for (i,flight) in enumerate(df_flight.flight)
    flight in (:Flt1007,) || continue
    df_flight.xyz_file[i] = MagNav.sgl_2020_train(flight)
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

flight = :Flt1007
line   = 1007.06                      # paper's primary line, full length
@info("loading $flight")
xyz  = get_XYZ(flight,df_flight;silent=true)
ind  = get_ind(xyz,line,df_nav)
map_name = df_nav[(df_nav.flight.==flight).&(df_nav.line.==line),:map_name][1]
mapS = get_map(map_name,df_map)

traj = get_traj(xyz,ind)
ins  = get_ins( xyz,ind;N_zero_ll=1)
(map_val,itp_mapS) = get_map_val(mapS,traj;return_itp=true)
flux = xyz.flux_d(ind)
N    = traj.N
println("line $line: N=$N, ~$(round(N*traj.dt/60,digits=1)) min, map=$map_name")

# horizontal position DRMS [m], evaluated after a 10-min warm-up (paper convention)
function drms(fo; warm=600.0)
    m  = (traj.tt .- traj.tt[1]) .>= warm
    dn = dlat2dn.(fo.lat[m] .- traj.lat[m], traj.lat[m])
    de = dlon2de.(fo.lon[m] .- traj.lon[m], traj.lat[m])
    sqrt(mean(dn.^2 .+ de.^2))
end
ins_drms = let m=(traj.tt.-traj.tt[1]).>=600.0
    sqrt(mean(dlat2dn.(ins.lat[m].-traj.lat[m],traj.lat[m]).^2 .+
              dlon2de.(ins.lon[m].-traj.lon[m],traj.lat[m]).^2)) end

results = DataFrame(method=String[],mag=String[],drms=Float64[])
log!(name,mag,fo) = (d=drms(fo); push!(results,(name,mag,d));
    println(rpad(name,26)," ",mag," DRMS(>10min) = ",round(d,digits=1)," m"); d)

##* cold-start TL model (generic init, no calibration flight)
n_TL     = 19                                    # perm3+ind6+eddy9+bias1
x0_TL    = zeros(n_TL)
TL_sigma = fill(1.0, n_TL)
P0_TL    = Matrix(Diagonal(fill(1.0, n_TL)))
(P0,Qd,R) = create_model(traj.dt,traj.lat[1];
                         init_pos_sigma=0.1,init_alt_sigma=1.0,init_vel_sigma=1.0,
                         meas_var=5^2,fogm_sigma=3,fogm_tau=180,
                         vec_states=false,TL_sigma=TL_sigma,P0_TL=P0_TL)
(P0n,Qdn,Rn) = create_model(traj.dt,traj.lat[1];
                            init_pos_sigma=0.1,init_alt_sigma=1.0,init_vel_sigma=1.0,
                            meas_var=5^2,fogm_sigma=3,fogm_tau=180)

println("\nINS (no aiding) DRMS(>10min) = ",round(ins_drms,digits=1)," m\n")

# stinger reference: EKF on the pre-compensated Mag 1 (only mag_1_c exists;
# cabin mags 2-5 are uncompensated only)
try
    fo = run_filt(traj,ins,xyz.mag_1_c[ind],itp_mapS,:ekf;P0=P0n,Qd=Qdn,R=Rn,
                  core=true,run_crlb=false); log!("EKF (Mag1 compensated)","Mag 1",fo)
catch e; @warn("EKF Mag1 failed",e) end
println()

# OUR batch FGO with TL factor nodes, cold start (uncompensated cabin mags)
for magsym in (:mag_4_uc, :mag_5_uc)
    mag_uc = getfield(xyz,magsym)[ind]
    tag    = replace(String(magsym),"mag_"=>"Mag ","_uc"=>"")
    println("=== $tag (uncompensated cabin magnetometer, cold start) ===")
    try  # full-batch (static TL) — degrades over the long flight
        fo = run_filt(traj,ins,mag_uc,itp_mapS,:fgo_online;P0=P0,Qd=Qd,R=R,
                      flux=flux,x0_TL=x0_TL,core=true,run_crlb=false)
        log!("FGO-online batch (static TL)",tag,fo)
    catch e; @warn("fgo_online batch failed",e) end
    try  # fixed-lag sliding window — TL adapts over the flight (iSAM2-style)
        frw = fgo_online(ins,mag_uc,flux,itp_mapS,x0_TL,P0,Qd,R;
                         win=300.0,overlap=90.0,core=true)
        log!("FGO-online win 5min (adaptive TL)",tag,MagNav.eval_filt(traj,ins,frw))
        frw = fgo_online(ins,mag_uc,flux,itp_mapS,x0_TL,P0,Qd,R;
                         win=120.0,overlap=45.0,core=true)
        log!("FGO-online win 2min (adaptive TL)",tag,MagNav.eval_filt(traj,ins,frw))
        frw = fgo_online(ins,mag_uc,flux,itp_mapS,x0_TL,P0,Qd,R;
                         win=300.0,overlap=90.0,robust=:huber,core=true)
        log!("FGO-online win 5min +Huber",tag,MagNav.eval_filt(traj,ins,frw))
    catch e; @warn("fgo_online window failed",e) end
    println()
end

##* paper's PUBLISHED cold-start DRMS on line 1007.06 (Hager et al. 2026),
##* reproduced here as a CITED reference only — NOT re-run in this script.
paper = DataFrame(mag=["Mag 1","Mag 3","Mag 4","Mag 5"],
                  paper_TL_only=[17.0,46.0,58.0,15.0],
                  paper_TL_NN  =[17.0,42.0,37.0,14.0])

println("=== OUR results (DRMS after 10-min warm-up, line $line, full length) ===")
show(results;allrows=true,allcols=true); println()
println("\n=== paper reported cold-start DRMS [m] (Hager et al. 2026, CITED — not re-run) ===")
show(paper;allrows=true,allcols=true); println()
CSV.write(joinpath(@__DIR__,"paper_baseline_results.csv"),results)
CSV.write(joinpath(@__DIR__,"paper_reference_values.csv"),paper)
println("\nNOTE: paper values are their PUBLISHED cold-start results. Our online",
        " EKF+TL+NN re-run of that filter family is in research/paper_impl.jl",
        " (reproduced Mag 4 40.0 m, Mag 5 17.5 m — in the paper's band).")
