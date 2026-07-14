##* Monte-Carlo consistency & significance study (statistical rigor pass)
#
# The real SGL flight lines are each a single physical realization, so a classical
# Monte-Carlo over measurement-noise draws is only valid in SIMULATION. Here we
# fix ONE simulated trajectory over a real anomaly map and, for each of N seeds,
# re-draw an independent INS-error realization (create_ins) and an independent
# clean scalar-measurement realization (create_mag_c). On every draw we run the
# same navigation model through a causal EKF and the batch factor graph (FGO), on
# the clean/compensated measurement so the comparison isolates the estimator (this
# mirrors the batch-estimator role of real line 1003.02, and by Lemma 1 the
# fixed-lag window shares the batch MAP in the linear-Gaussian case).
#
# Outputs:
#   - DRMS mean +/- 95% CI and the paired win-rate P(FGO < EKF)          (Test A)
#   - ANEES on 2-DOF horizontal position, vs the chi-square consistency
#     band, for EKF and FGO (is the reported covariance credible?)       (Test B)
#   - per-epoch NEES time series and a +/-2 sigma envelope trace from one
#     representative seed, for the figures                            (Test B/C)
#
# NEES is computed coordinate-free in the state's native (rad) position units:
#   eps_t = e_pos,t' * inv(P[1:2,1:2,t]) * e_pos,t,   E[eps]=2 (2 DOF).
#
# Usage: julia --project=. research/fgo_montecarlo.jl [N]

using MagNav
using CSV, DataFrames
using LinearAlgebra, Statistics
using Random: seed!

const N_MC   = length(ARGS) >= 1 ? parse(Int,ARGS[1]) : 30
const WARM   = 30.0          # [s] warm-up excluded from DRMS/NEES (clean sim)
const MEASV  = 1.0^2         # scalar measurement variance [nT^2]
const FOGM_S = 1.0
const FOGM_T = 600.0
const IPOS   = 3.0           # init position sigma [m] (P0 and create_ins matched)
const IVEL   = 0.01
const IALT   = 0.001

# chi-square 95% two-sided quantiles (hardcoded, no Distributions dep)
const CHI2_2_LO, CHI2_2_HI = 0.05064, 7.37776   # 2 DOF (per-epoch NEES band)

##* one simulated trajectory over a real high-resolution map --------------------
seed!(20)   # fix the trajectory draw so the reported numbers are reproducible
try
    df_map = DataFrame(CSV.File(joinpath(@__DIR__,"..","examples","dataframes","df_map.csv")))
    df_map[!,:map_name] = Symbol.(df_map[!,:map_name])
    df_map[!,:map_file] = String.(df_map[!,:map_file])
    for i in eachindex(df_map.map_name)
        df_map.map_name[i] == :Eastern_395 || continue
        df_map.map_file[i] = MagNav.ottawa_area_maps(df_map.map_name[i])
    end
    global mapS = get_map(:Eastern_395,df_map)
    global map_used = "Eastern_395 (high-res survey)"
    global xyz  = create_XYZ0(mapS; alt=mapS.alt, dt=0.1, t=240, v=68,
                              N_waves=2, attempts=500, cor_sigma=1.0,
                              fogm_sigma=1.0, silent=true)
catch e
    @warn("Eastern_395 sim failed, falling back to NAMAD",e)
    global mapS = get_map(MagNav.namad)
    global map_used = "NAMAD (coarse)"
    global xyz  = create_XYZ0(mapS; alt=1000, dt=0.1, t=600, v=68, N_waves=3,
                              cor_sigma=1.0, fogm_sigma=1.0, silent=true)
end
traj = xyz.traj
N    = traj.N
itp  = map_interpolate(mapS)
println("map: $map_used | N=$N epochs | dt=$(traj.dt) s | N_MC=$N_MC seeds")

# estimator model, matched to the draw distributions above
(P0,Qd,R) = create_model(traj.dt,traj.lat[1];
    init_pos_sigma=IPOS, init_alt_sigma=IALT, init_vel_sigma=IVEL,
    meas_var=MEASV, fogm_sigma=FOGM_S, fogm_tau=FOGM_T)

mask = (traj.tt .- traj.tt[1]) .>= WARM

# run_filt returns a FILTout with metre-space position error (n_err,e_err) and the
# estimator +/-1 sigma (n_std,e_std) already mapped to metres from the covariance.
drms_of(fo) = sqrt(mean((fo.n_err.^2 .+ fo.e_err.^2)[mask]))

# per-axis (diagonal) 2-DOF horizontal-position NEES; each term is a squared
# standard normal under a consistent covariance, so E[NEES]=2 and it is compared
# to the chi-square(2) band. Uses the diagonal std (cross-covariance not exposed
# by FILTout); reported as a diagonal normalization.
nees_series(fo) = (fo.n_err ./ fo.n_std).^2 .+ (fo.e_err ./ fo.e_std).^2

##* Monte-Carlo loop -----------------------------------------------------------
# Three map-matching estimators on the clean compensated sensor: the causal EKF,
# the marginalized (Rao-Blackwellized) particle filter already in MagNav.jl, and
# the batch factor graph. The MPF is a map-matching filter with no online-TL
# compensation, so it belongs to this clean-sensor comparison, not the cold-start
# breadth (where it would diverge like the plain EKF).
const NUM_PART = 1000
drms_ekf = Float64[]; drms_mpf = Float64[]; drms_fgo = Float64[]
nees_ekf_all = Float64[]; nees_mpf_all = Float64[]; nees_fgo_all = Float64[]
nfail = 0
rep_saved = false
rep_nees_ekf = Float64[]; rep_nees_mpf = Float64[]; rep_nees_fgo = Float64[]
rep_sigma = (Float64[],Float64[],Float64[],Float64[])

for s = 1:N_MC
    seed!(1000 + s)
    ins = create_ins(traj; init_pos_sigma=IPOS, init_alt_sigma=IALT,
                     init_vel_sigma=IVEL)
    mag = MagNav.create_mag_c(traj.lat, traj.lon, mapS; alt=traj.alt[1], dt=traj.dt,
                       meas_var=MEASV, fogm_sigma=FOGM_S, fogm_tau=FOGM_T,
                       silent=true)
    local fe, fm, ff
    try
        fe = run_filt(traj,ins,mag,itp,:ekf;P0=P0,Qd=Qd,R=R,core=false,run_crlb=false)
        fm = run_filt(traj,ins,mag,itp,:mpf;P0=P0,Qd=Qd,R=R,core=false,
                      num_part=NUM_PART,run_crlb=false)
        ff = run_filt(traj,ins,mag,itp,:fgo;P0=P0,Qd=Qd,R=R,core=false,run_crlb=false)
    catch e
        global nfail += 1; @warn("seed $s failed",e); continue
    end
    push!(drms_ekf, drms_of(fe)); push!(drms_mpf, drms_of(fm))
    push!(drms_fgo, drms_of(ff))
    ne = nees_series(fe); nm = nees_series(fm); nf = nees_series(ff)
    append!(nees_ekf_all, ne[mask]); append!(nees_mpf_all, nm[mask])
    append!(nees_fgo_all, nf[mask])
    if !rep_saved
        global rep_nees_ekf = ne; global rep_nees_mpf = nm; global rep_nees_fgo = nf
        global rep_sigma = (ff.n_err, ff.e_err, ff.n_std, ff.e_std)
        global rep_saved = true
    end
    println("seed $s  EKF=$(round(drms_ekf[end],digits=1))  ",
            "MPF=$(round(drms_mpf[end],digits=1))  ",
            "FGO=$(round(drms_fgo[end],digits=1)) m")
end

nrun = length(drms_ekf)
nrun == 0 && error("all Monte-Carlo seeds failed")

ci95(v) = 1.96*std(v)/sqrt(length(v))                       # normal-approx 95% CI
anees(v) = mean(v)
# ANEES 95% band over M samples of a 2-DOF NEES (mean 2, var 4): 2 +/- 1.96*sqrt(4/M)
aneesband(M) = (2 - 1.96*sqrt(4/M), 2 + 1.96*sqrt(4/M))

winrate = mean(drms_fgo .< drms_ekf)
(alo,ahi) = aneesband(length(nees_ekf_all))

summ = DataFrame(
    estimator = ["EKF (causal)","MPF (particle)","FGO (batch)"],
    drms_mean = round.([mean(drms_ekf), mean(drms_mpf), mean(drms_fgo)],digits=2),
    drms_ci95 = round.([ci95(drms_ekf), ci95(drms_mpf), ci95(drms_fgo)],digits=2),
    drms_std  = round.([std(drms_ekf),  std(drms_mpf),  std(drms_fgo)],digits=2),
    anees     = round.([anees(nees_ekf_all), anees(nees_mpf_all),
                        anees(nees_fgo_all)],digits=3),
    anees_lo  = round(alo,digits=3),
    anees_hi  = round(ahi,digits=3),
)

println("\n=== Monte-Carlo consistency (N=$nrun successful seeds, $nfail failed) ===")
show(summ;allrows=true,allcols=true); println()
println("FGO beats EKF (DRMS) on $(round(100*winrate,digits=0))% of seeds")
println("Per-epoch 2-DOF NEES 95% band [$(CHI2_2_LO), $(CHI2_2_HI)]; ",
        "ANEES ideal 2.0, 95% band over $(length(nees_ekf_all)) samples ",
        "[$(round(alo,digits=3)), $(round(ahi,digits=3))]")

##* write CSVs for the figures/tables ------------------------------------------
CSV.write(joinpath(@__DIR__,"montecarlo_summary.csv"), summ)

tt = traj.tt .- traj.tt[1]
CSV.write(joinpath(@__DIR__,"montecarlo_nees.csv"),
    DataFrame(t=tt, nees_ekf=rep_nees_ekf, nees_mpf=rep_nees_mpf,
              nees_fgo=rep_nees_fgo))
(n_err,e_err,n_std,e_std) = rep_sigma
CSV.write(joinpath(@__DIR__,"montecarlo_sigma.csv"),
    DataFrame(t=tt, n_err=n_err, e_err=e_err, n_std=n_std, e_std=e_std))
println("\nwrote montecarlo_summary.csv, montecarlo_nees.csv, montecarlo_sigma.csv")
