##* Consistency (NEES) fairness: does the MPF over-confidence survive a proper config?
#
# The paper consistency study runs the toolbox MPF (run_filt :mpf: raw-exp weights,
# cubic itp, default threshold, NO roughening) and reports ANEES=59.8 (severely
# over-confident, particle-depletion). On this CLEAN simulated sensor R is known
# exactly, so the only config issue is depletion. This re-runs the SAME 30-seed sim
# with the properly-configured RBPF (mpf_logw: log weights, linear itp, standard
# resampling thresh=0.5) and a roughening sweep, alongside the toolbox MPF, EKF, and
# FGO. If roughening pulls the ANEES down to ~2, the over-confidence is a config
# artifact (depletion the regularization removes), and the paper claim must become
# config-conditional; if it stays high, the claim holds.
#
# Outputs research/mc_nees_mpf_results.csv. Usage: julia --project=. research/mc_nees_mpf.jl [N_MC]

using MagNav
using CSV, DataFrames
using LinearAlgebra, Statistics
using Random: seed!

include(joinpath(@__DIR__,"mpf_logw.jl"))

const N_MC   = length(ARGS) >= 1 ? parse(Int,ARGS[1]) : 30
const WARM   = 30.0
const MEASV  = 1.0^2
const FOGM_S = 1.0
const FOGM_T = 600.0
const IPOS   = 3.0
const IVEL   = 0.01
const IALT   = 0.001
const NUM_PART = 1000
const ROUGHS   = [0.0, 1.0, 3.0]     # roughening sweep [m] for the proper-config MPF

seed!(20)
df_map = DataFrame(CSV.File(joinpath(@__DIR__,"..","examples","dataframes","df_map.csv")))
df_map[!,:map_name] = Symbol.(df_map[!,:map_name])
df_map[!,:map_file] = String.(df_map[!,:map_file])
for i in eachindex(df_map.map_name)
    df_map.map_name[i] == :Eastern_395 || continue
    df_map.map_file[i] = MagNav.ottawa_area_maps(df_map.map_name[i])
end
mapS = get_map(:Eastern_395,df_map)
xyz  = create_XYZ0(mapS; alt=mapS.alt, dt=0.1, t=240, v=68, N_waves=2, attempts=500,
                   cor_sigma=1.0, fogm_sigma=1.0, silent=true)
traj = xyz.traj
itp     = map_interpolate(mapS)           # cubic (toolbox MPF / EKF / FGO)
itp_lin = map_interpolate(mapS, :linear)  # linear (proper-config MPF)
println("Eastern_395 | N=$(traj.N) | N_MC=$N_MC")

(P0,Qd,R) = create_model(traj.dt,traj.lat[1];init_pos_sigma=IPOS,init_alt_sigma=IALT,
    init_vel_sigma=IVEL,meas_var=MEASV,fogm_sigma=FOGM_S,fogm_tau=FOGM_T)
mask = (traj.tt .- traj.tt[1]) .>= WARM
drms_of(fo) = sqrt(mean((fo.n_err.^2 .+ fo.e_err.^2)[mask]))
nees_series(fo) = (fo.n_err ./ fo.n_std).^2 .+ (fo.e_err ./ fo.e_std).^2

# accumulators: keys = method label
labels = ["EKF","FGO","MPF toolbox"]
for r in ROUGHS; push!(labels, "MPF logw+rgh$(r)"); end
drms = Dict(l=>Float64[] for l in labels)
nees = Dict(l=>Float64[] for l in labels)

for s = 1:N_MC
    seed!(1000 + s)
    ins = create_ins(traj; init_pos_sigma=IPOS, init_alt_sigma=IALT, init_vel_sigma=IVEL)
    mag = MagNav.create_mag_c(traj.lat, traj.lon, mapS; alt=traj.alt[1], dt=traj.dt,
                       meas_var=MEASV, fogm_sigma=FOGM_S, fogm_tau=FOGM_T, silent=true)
    try
        fe = run_filt(traj,ins,mag,itp,:ekf;P0=P0,Qd=Qd,R=R,core=false,run_crlb=false)
        ff = run_filt(traj,ins,mag,itp,:fgo;P0=P0,Qd=Qd,R=R,core=false,run_crlb=false)
        fm = run_filt(traj,ins,mag,itp,:mpf;P0=P0,Qd=Qd,R=R,core=false,num_part=NUM_PART,run_crlb=false)
        for (l,fo) in (("EKF",fe),("FGO",ff),("MPF toolbox",fm))
            push!(drms[l], drms_of(fo)); append!(nees[l], nees_series(fo)[mask])
        end
        for rgh in ROUGHS
            fr = mpf_logw(ins,mag,itp_lin;P0=P0,Qd=Qd,R=R,num_part=NUM_PART,
                          thresh=0.5,roughen_m=rgh,core=false)
            l = "MPF logw+rgh$(rgh)"
            if fr.c
                fo = MagNav.eval_filt(traj,ins,fr)
                push!(drms[l], drms_of(fo)); append!(nees[l], nees_series(fo)[mask])
            end
        end
    catch e; @warn("seed $s failed",e); continue end
    println("seed $s done")
end

results = DataFrame(method=String[],DRMS_mean=Float64[],DRMS_ci95=Float64[],ANEES=Float64[],n_seed=Int[])
for l in labels
    isempty(drms[l]) && continue
    dm = mean(drms[l]); ci = 1.96*std(drms[l])/sqrt(length(drms[l]))
    an = mean(nees[l])
    push!(results,(l,round(dm,digits=2),round(ci,digits=2),round(an,digits=2),length(drms[l])))
end

CSV.write(joinpath(@__DIR__,"mc_nees_mpf_results.csv"), results)
println("\n=== NEES fairness: toolbox vs proper-config MPF (ideal ANEES=2) ===")
show(results, allrows=true, allcols=true); println()
println("wrote mc_nees_mpf_results.csv")
