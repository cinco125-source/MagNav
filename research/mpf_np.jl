##* Particle-count check: does MPF -> EKF as N grows (finite-N MC variance)?
#
# At the matched config (R = 40 nT ~ offline-NN residual, roughening 5 m) the repaired
# native MPF recovered to ~EKF-class at N=1000 (1007.06: 40.9 vs EKF 33.3; 1003.08:
# 102 vs 41). Theory: MPF < EKF here is finite-particle Monte-Carlo variance on a
# near-linear-Gaussian problem, so DRMS should fall toward the EKF value as N grows.
# Run N in {1000, 4000} on the offline-NN compensated Mag 5, both lines, EKF reference.
#
# Outputs research/mpf_np_results.csv. Usage: julia --project=. research/mpf_np.jl

using MagNav
using CSV, DataFrames
using LinearAlgebra, Statistics
using Random: seed!
seed!(33)

include(joinpath(@__DIR__,"mpf_logw.jl"))

FOGM_SIG = 3.0
FOGM_TAU = 180.0
MEAS_STD = 40.0     # matched to the offline-NN residual (~36-50 nT)
ROUGH    = 5.0
NPS      = [1000, 4000]
LINES = [(:Flt1003,1003.08), (:Flt1007,1007.06)]
TRAIN_FLIGHTS = [:Flt1002,:Flt1006]
flights = unique(vcat(first.(LINES),TRAIN_FLIGHTS))

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
df_all = DataFrame(CSV.File(joinpath(df_dir,"df_all.csv")))
df_all[!,:flight] = Symbol.(df_all[!,:flight])

xyz_cache = Dict{Symbol,Any}()
getxyz(fl) = get!(()->get_XYZ(fl,df_flight;silent=true), xyz_cache, fl)

function drms_ll(traj, lat, lon; warm=600.0, div_thresh=1e4)
    m  = (traj.tt .- traj.tt[1]) .>= warm
    dn = dlat2dn.(lat[m] .- traj.lat[m], traj.lat[m])
    de = dlon2de.(lon[m] .- traj.lon[m], traj.lat[m])
    d  = sqrt(mean(dn.^2 .+ de.^2))
    (isfinite(d) && d < div_thresh) ? d : Inf
end

FEATURES = [:TL_A_flux_d,:cur_com_1,:cur_ac_lo,:cur_tank,:cur_strb,:cur_heat,
            :vol_bat_1,:vol_bat_2]

train_lines = df_all[in.(df_all.flight,Ref(TRAIN_FLIGHTS)),:line]
println("training frozen offline :m1 for mag_5_uc on $(TRAIN_FLIGHTS)")
cp = MagNav.NNCompParams(features_setup=FEATURES, model_type=:m1, y_type=:d,
                         use_mag=:mag_5_uc, use_vec=:flux_d,
                         epoch_adam=20, hidden=[8], batchsize=2048)
(model5,_,_,_,_) = comp_train(cp, train_lines, df_all, df_flight, df_map;silent=true)

results = DataFrame(flight=Symbol[],line=Float64[],comp_rmse_nT=Float64[],
                    num_part=Int[],EKF=Float64[],MPF=Float64[],t_s=Float64[])

for (fl,line) in LINES
    xyz  = getxyz(fl)
    ind  = get_ind(xyz,line,df_nav)
    mname= df_nav[(df_nav.flight.==fl).&(df_nav.line.==line),:map_name][1]
    mapS = get_map(mname,df_map)
    traj = get_traj(xyz,ind)
    ins  = get_ins(xyz,ind;N_zero_ll=1)
    itp_lin = map_interpolate(mapS, :linear)
    ref  = xyz.mag_1_c[ind]
    mag  = xyz.mag_5_uc[ind]
    (_,y_hat,_,_) = comp_test(model5, xyz, ind; silent=true)
    mag_nn = mag .- y_hat
    crms = sqrt(mean((mag_nn .- ref).^2))

    (P0,Qd,R) = create_model(traj.dt,traj.lat[1];init_pos_sigma=3.0,init_alt_sigma=1.0,
        init_vel_sigma=1.0,meas_var=MEAS_STD^2,fogm_sigma=FOGM_SIG,fogm_tau=FOGM_TAU)
    d_ekf = try
        drms_ll(traj,(fo=run_filt(traj,ins,mag_nn,itp_lin,:ekf;P0=P0,Qd=Qd,R=R,
                                  core=true,run_crlb=false)).lat,fo.lon)
    catch e; @warn("ekf",e); Inf end
    println("\n$fl $line ($mname)  comp-RMSE=$(round(crms,digits=1)) nT  EKF=$(round(d_ekf,digits=1)) m")

    for np in NPS
        t = @elapsed (d_mpf = try
            fr = mpf_logw(ins,mag_nn,itp_lin;P0=P0,Qd=Qd,R=R,num_part=np,
                          thresh=0.1,roughen_m=ROUGH,core=true)
            fr.c ? drms_ll(traj,(fo=MagNav.eval_filt(traj,ins,fr)).lat,fo.lon) : Inf
        catch e; @warn("mpf_logw",e); Inf end)
        push!(results,(fl,line,round(crms,digits=1),np,round(d_ekf,digits=1),
                       round(d_mpf,digits=1),round(t,digits=1)))
        println("  N=$np   MPF=$(round(d_mpf,digits=1)) m   ($(round(t,digits=0)) s)")
    end
end

CSV.write(joinpath(@__DIR__,"mpf_np_results.csv"), results)
println("\n=== particle-count check: offline-NN Mag 5, matched R=40 nT + roughen 5 m ===")
show(results, allrows=true, allcols=true); println()
println("wrote mpf_np_results.csv")
