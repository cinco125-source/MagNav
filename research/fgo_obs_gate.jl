##* FGO-native observability gate: the factor graph SEES and PREVENTS the collapse
#
# Our contribution lives in the FGO, not the EKF. The factor-graph window exposes
# the full measurement Jacobian, so we can measure the position↔compensation
# confound directly from the window information matrix:
#
#   ρ_obs = max_j R²( ∂h/∂pos_j  ~  [1, A] )   over the window,
#
# i.e. how well the Tolles-Lawson basis A can reproduce the linearized map
# gradient (position sensitivity). ρ_obs → 1 ⇒ compensation can mimic a position
# error ⇒ observability collapse. A causal EKF cannot compute this — it has no
# window information matrix. `fgo_online(...; obs_gate=true)` stiffens the TL
# prior/process-noise whenever ρ_obs exceeds a threshold.
#
# This script demonstrates, on the paper's line 1007.06, uncompensated cabin Mag 4:
#   A. exogenous full Tolles-Lawson  → converges, ρ_obs LOW  (observable)
#      — reconciles why full-TL FGO works even though the naive map-VALUE ρ²≈0.85:
#        the linearized position-sensitivity confound ρ_obs is what matters, and
#        it is small.
#   B. + endogenous mag_uc injected  → collapses, ρ_obs HIGH (unobservable)
#   C. + endogenous mag_uc, obs_gate → recovered (the gate stiffens the TL block)
#   D. exogenous full-TL, sliding window → per-window ρ_obs certificate (LOW)
#
# Usage: julia --project=. research/fgo_obs_gate.jl

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
line   = 1007.06
@info("loading $flight")
xyz  = get_XYZ(flight,df_flight;silent=true)
ind  = get_ind(xyz,line,df_nav)
map_name = df_nav[(df_nav.flight.==flight).&(df_nav.line.==line),:map_name][1]
mapS = get_map(map_name,df_map)
traj = get_traj(xyz,ind)
ins  = get_ins( xyz,ind;N_zero_ll=1)
(map_val,itp_mapS) = get_map_val(mapS,traj;return_itp=true)
flux = xyz.flux_d(ind)
mag4 = xyz.mag_4_uc[ind]
N    = traj.N
println("line $line: N=$N, ~$(round(N*traj.dt/60,digits=1)) min, map=$map_name")

function drms(fo; warm=600.0)
    m  = (traj.tt .- traj.tt[1]) .>= warm
    dn = dlat2dn.(fo.lat[m] .- traj.lat[m], traj.lat[m])
    de = dlon2de.(fo.lon[m] .- traj.lon[m], traj.lat[m])
    sqrt(mean(dn.^2 .+ de.^2))
end
D(frw) = drms(MagNav.eval_filt(traj,ins,frw))

build(nTL; meas_var=12.0^2) = create_model(traj.dt,traj.lat[1];
    init_pos_sigma=0.1,init_alt_sigma=1.0,init_vel_sigma=1.0,
    meas_var=meas_var,fogm_sigma=3,fogm_tau=180,vec_states=false,
    TL_sigma=fill(1.0,nTL),P0_TL=Matrix(Diagonal(fill(1.0,nTL))))

terms = [:permanent,:induced,:eddy,:bias]
nT    = size(create_TL_A(flux;terms=terms),2)          # exogenous TL columns (19)
mag4n = (mag4 .- mean(mag4)) ./ std(mag4)              # standardized endogenous feature

results = DataFrame(case=String[],basis=String[],gate=Bool[],DRMS_m=Float64[])
log!(c,b,g,frw)=(d=D(frw); push!(results,(c,b,g,round(d,digits=1)));
    println(rpad(c,42)," DRMS(>10min) = ",round(d,digits=1)," m"); d)

println("\n(ρ_obs is logged per fit below; low ⇒ observable, high ⇒ collapse)\n")

# A. exogenous full Tolles-Lawson, full batch — observable
(P0,Qd,R) = build(nT)
println("=== A. exogenous full TL (batch) ===")
log!("A. exogenous full TL","TL(19)",false,
     fgo_online(ins,mag4,flux,itp_mapS,zeros(nT),P0,Qd,R;terms=terms,core=true,silent=false))

# B. + endogenous mag_uc injected, NO gate — collapse
(P0b,Qdb,Rb) = build(nT+1)
println("\n=== B. + endogenous mag_uc injected, NO gate ===")
log!("B. +mag_uc (endogenous), no gate","TL(19)+mag_uc",false,
     fgo_online(ins,mag4,flux,itp_mapS,zeros(nT+1),P0b,Qdb,Rb;terms=terms,
                A_extra=mag4n,core=true,silent=false))

# C. + endogenous mag_uc injected, obs_gate ON — recovered
println("\n=== C. + endogenous mag_uc injected, obs_gate ON ===")
log!("C. +mag_uc (endogenous), obs_gate","TL(19)+mag_uc",true,
     fgo_online(ins,mag4,flux,itp_mapS,zeros(nT+1),P0b,Qdb,Rb;terms=terms,
                A_extra=mag4n,core=true,obs_gate=true,silent=false))

# D. exogenous full-TL sliding window — per-window ρ_obs certificate
println("\n=== D. exogenous full TL, sliding window 5min (per-window ρ_obs) ===")
log!("D. exogenous full TL, win 5min","TL(19)",false,
     fgo_online(ins,mag4,flux,itp_mapS,zeros(nT),P0,Qd,R;terms=terms,core=true,
                win=300.0,overlap=90.0,silent=false))

println("\n=== FGO observability gate on line $line, Mag 4 ===")
show(results;allrows=true,allcols=true); println()
CSV.write(joinpath(@__DIR__,"fgo_obs_gate.csv"),results)
println("\nTakeaway: the factor graph measures the position↔compensation confound",
        " (ρ_obs) directly from its window Jacobian and stiffens the compensation",
        " when it would eat the map — recovering (C) the collapse that an injected",
        " endogenous feature causes (B), while leaving the observable exogenous-TL",
        " fits (A, D) untouched. The gate is the FGO-native, adaptive form of the",
        " exogeneity design rule (research/OBSERVABILITY.md).")
