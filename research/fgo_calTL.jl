##* Calibration-flight TL baseline: the workflow the dataset was designed around.
#
# SGL 2020 ships dedicated high-altitude compensation maneuvers (df_cal.csv):
# the Flt1002 cloverleaf (1002.20) and square (1002.02) patterns. The classical
# workflow fits Tolles-Lawson there, freezes the coefficients, and navigates.
# This script runs that workflow on the cabin magnetometers over the four counted
# lines and reports (a) the residual-interference RMSE on each line (comparable to
# Laoue 2022, who measured 51-135 nT for these sensors) and (b) the navigation
# DRMS through the standard EKF and the batch FGO -- the missing anchor against
# which the cold-start joint estimator should be judged. Calibration and
# navigation are different flights/days/altitudes, so this also measures how well
# a high-altitude calibration transfers.
#
# Outputs research/fgo_calTL_results.csv.
# Usage: julia --project=. research/fgo_calTL.jl

using MagNav
using CSV, DataFrames
using LinearAlgebra, Statistics
using Random: seed!
seed!(33)

MEAS_VAR = 12.0^2         # paper protocol measurement variance
FOGM_SIG = 3.0
FOGM_TAU = 180.0
TERMS    = [:permanent,:induced,:eddy]   # create_TL_coef default (no bias)
LINES = [(:Flt1003,1003.02), (:Flt1003,1003.08),
         (:Flt1007,1007.02), (:Flt1007,1007.06)]
flights = unique(first.(LINES))

df_dir    = joinpath(@__DIR__,"..","examples","dataframes")
df_flight = DataFrame(CSV.File(joinpath(df_dir,"df_flight.csv")))
df_flight[!,:flight]   = Symbol.(df_flight[!,:flight])
df_flight[!,:xyz_type] = Symbol.(df_flight[!,:xyz_type])
df_flight[!,:xyz_file] = String.(df_flight[!,:xyz_file])
for (i,fl) in enumerate(df_flight.flight)
    (fl in flights || fl == :Flt1002) || continue
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
df_cal = DataFrame(CSV.File(joinpath(df_dir,"df_cal.csv")))
df_cal[!,:flight] = Symbol.(df_cal[!,:flight])

xyz_cache = Dict{Symbol,Any}()
getxyz(fl) = get!(()->get_XYZ(fl,df_flight;silent=true), xyz_cache, fl)

function drms_ll(traj, lat, lon; warm=600.0, div_thresh=1e4)
    m  = (traj.tt .- traj.tt[1]) .>= warm
    dn = dlat2dn.(lat[m] .- traj.lat[m], traj.lat[m])
    de = dlon2de.(lon[m] .- traj.lon[m], traj.lat[m])
    d  = sqrt(mean(dn.^2 .+ de.^2))
    (isfinite(d) && d < div_thresh) ? d : Inf
end

##* fit TL on the Flt1002 compensation maneuvers ---------------------------------
xyz2 = getxyz(:Flt1002)
cal_ind(line) = begin
    rows = df_cal[(df_cal.flight.==:Flt1002).&(df_cal.line.==line),:]
    ind = falses(length(xyz2.traj.lat))
    for r in eachrow(rows)
        ind .|= get_ind(xyz2;tt_lim=[r.t_start,r.t_end])
    end
    ind
end
MANEUVERS = [("cloverleaf",cal_ind(1002.20)), ("square",cal_ind(1002.02))]
LAMBDAS   = [0.0, 0.025]

# coefs[(maneuver,lambda,magsym)] = (coef, fit_var)
coefs = Dict{Tuple{String,Float64,Symbol},Tuple{Vector{Float64},Float64}}()
for (mname,ind) in MANEUVERS, lam in LAMBDAS, magsym in (:mag_4_uc,:mag_5_uc)
    (c,v) = create_TL_coef(xyz2.flux_d, getfield(xyz2,magsym), ind;
                           terms=TERMS, λ=lam, return_var=true)
    coefs[(mname,lam,magsym)] = (Float64.(c), Float64(v))
    println("fit $(mname) λ=$(lam) $(magsym): fit-σ=$(round(sqrt(v),digits=1)) nT")
end

##* apply frozen coefficients on the counted lines -------------------------------
results = DataFrame(flight=Symbol[],line=Float64[],mag=String[],maneuver=String[],
                    lambda=Float64[],comp_rmse_nT=Float64[],
                    EKF_R12=Float64[],EKF_Rfit=Float64[],FGO_Rfit=Float64[])

for (fl,line) in LINES
    xyz   = getxyz(fl)
    ind   = get_ind(xyz,line,df_nav)
    mname = df_nav[(df_nav.flight.==fl).&(df_nav.line.==line),:map_name][1]
    mapS  = get_map(mname,df_map)
    traj  = get_traj(xyz,ind)
    ins   = get_ins(xyz,ind;N_zero_ll=1)
    (_,itp) = get_map_val(mapS,traj;return_itp=true)
    flux  = xyz.flux_d(ind)
    A     = create_TL_A(flux;terms=TERMS)
    ref   = xyz.mag_1_c[ind]           # compensated stinger, comp-quality proxy
    println("\n$fl $line ($mname)")
    for magsym in (:mag_4_uc,:mag_5_uc)
        mag = getfield(xyz,magsym)[ind]
        tag = replace(String(magsym),"mag_"=>"Mag ","_uc"=>"")
        for (mv,_) in MANEUVERS, lam in LAMBDAS
            (c,v) = coefs[(mv,lam,magsym)]
            mag_c = mag .- MagNav.detrend(A*c;mean_only=true)
            crms  = sqrt(mean((mag_c .- ref).^2))
            d12 = dfit = dfgo = Inf
            for (Rv,which) in ((MEAS_VAR,:R12),(max(v,1.0),:Rfit))
                (P0,Qd,R) = create_model(traj.dt,traj.lat[1];init_pos_sigma=0.1,
                    init_alt_sigma=1.0,init_vel_sigma=1.0,meas_var=Rv,
                    fogm_sigma=FOGM_SIG,fogm_tau=FOGM_TAU)
                try
                    fo = run_filt(traj,ins,mag_c,itp,:ekf;P0=P0,Qd=Qd,R=R,
                                  core=true,run_crlb=false)
                    dd = drms_ll(traj,fo.lat,fo.lon)
                    which == :R12 ? (d12 = dd) : (dfit = dd)
                catch e; @warn("EKF $fl $line $tag $mv λ=$lam $which failed",e) end
                if which == :Rfit
                    try
                        fo = run_filt(traj,ins,mag_c,itp,:fgo;P0=P0,Qd=Qd,R=R,
                                      core=true,run_crlb=false,robust=:huber)
                        dfgo = drms_ll(traj,fo.lat,fo.lon)
                    catch e; @warn("FGO $fl $line $tag $mv λ=$lam failed",e) end
                end
            end
            push!(results,(fl,line,tag,mv,lam,round(crms,digits=1),
                           round(d12,digits=1),round(dfit,digits=1),
                           round(dfgo,digits=1)))
            println("  $tag $(rpad(mv,10)) λ=$(rpad(lam,6)) comp-RMSE=$(round(crms,digits=1)) nT",
                    "  EKF(R12)=$(round(d12,digits=1))  EKF(Rfit)=$(round(dfit,digits=1))",
                    "  FGO(Rfit)=$(round(dfgo,digits=1)) m")
        end
    end
end

println("\n=== calibration-flight TL baseline (frozen cloverleaf/square coefficients) ===")
show(results;allrows=true,allcols=true); println()
CSV.write(joinpath(@__DIR__,"fgo_calTL_results.csv"),results)
println("wrote fgo_calTL_results.csv")
