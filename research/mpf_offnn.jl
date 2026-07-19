##* Fixed offline-NN compensation + the REPAIRED native MPF (no online estimation).
#
# The user's control: strip the interference with a FROZEN, offline-trained NN so the
# measurement is genuinely accurate, then hand the clean signal to the native MPF for
# navigation ONLY (no online TL, no online NN). If the MPF's failure were merely SNR /
# measurement accuracy, an accurate offline-compensated signal should let it navigate.
#
# comp_grid.jl already has an "MPF offNN" cell, but it uses run_filt(:mpf) -- the
# raw-exp / cubic-itp / thresh-0.8 MPF that underflows (its DRMS is identical across
# Mag 4/5 per line: the measurement update did nothing, it dead-reckoned the INS). This
# script feeds the SAME frozen offline-NN signal to the repaired native MPF
# (mpf_logw: log-domain weights, linear itp, low resample threshold, roughening) and
# to a plain EKF as the estimator reference on the identical measurement.
#
#   compensation: frozen offline NN (:m1, trained on Flt1002+1006, applied cross-flight)
#   estimator:    EKF (offNN)  vs  repaired native MPF (offNN)
#
# Outputs research/mpf_offnn_results.csv. Usage: julia --project=. research/mpf_offnn.jl

using MagNav
using CSV, DataFrames
using LinearAlgebra, Statistics
using Random: seed!
seed!(33)

include(joinpath(@__DIR__,"mpf_logw.jl"))   # repaired native MPF (log weights + roughening)

MEAS_VAR = 12.0^2
FOGM_SIG = 3.0
FOGM_TAU = 180.0
NP       = 1000
THR      = 0.1
RGH      = 5.0
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

##* train one frozen offline :m1 model per cabin magnetometer -------------------
models = Dict{Symbol,Any}()
train_lines = df_all[in.(df_all.flight,Ref(TRAIN_FLIGHTS)),:line]
for magsym in (:mag_4_uc,:mag_5_uc)
    println("training frozen offline :m1 for $magsym on $(TRAIN_FLIGHTS)")
    cp = MagNav.NNCompParams(features_setup=FEATURES, model_type=:m1, y_type=:d,
                             use_mag=magsym, use_vec=:flux_d,
                             epoch_adam=20, hidden=[8], batchsize=2048)
    (cp_out,_,_,_,_) = comp_train(cp, train_lines, df_all, df_flight, df_map;silent=true)
    models[magsym] = cp_out
end

results = DataFrame(flight=Symbol[],line=Float64[],mag=String[],
                    comp_rmse_nT=Float64[],EKF_offNN=Float64[],MPF_offNN=Float64[])

for (fl,line) in LINES
    xyz  = getxyz(fl)
    ind  = get_ind(xyz,line,df_nav)
    mname= df_nav[(df_nav.flight.==fl).&(df_nav.line.==line),:map_name][1]
    mapS = get_map(mname,df_map)
    traj = get_traj(xyz,ind)
    ins  = get_ins(xyz,ind;N_zero_ll=1)
    itp_lin = map_interpolate(mapS, :linear)
    ref  = xyz.mag_1_c[ind]
    (P0,Qd,R) = create_model(traj.dt,traj.lat[1];init_pos_sigma=3.0,init_alt_sigma=1.0,
        init_vel_sigma=1.0,meas_var=MEAS_VAR,fogm_sigma=FOGM_SIG,fogm_tau=FOGM_TAU)

    println("\n$fl $line ($mname)")
    for magsym in (:mag_4_uc,:mag_5_uc)
        mag = getfield(xyz,magsym)[ind]
        tag = replace(String(magsym),"mag_"=>"Mag ","_uc"=>"")

        mag_nn = fill(NaN,length(mag))
        try
            (_,y_hat,_,_) = comp_test(models[magsym], xyz, ind; silent=true)
            mag_nn = mag .- y_hat
        catch e; @warn("comp_test failed $fl $line $tag",e) end
        crms = any(isnan,mag_nn) ? NaN : sqrt(mean((mag_nn .- ref).^2))

        d_ekf = try
            any(isnan,mag_nn) ? NaN :
            drms_ll(traj,(fo=run_filt(traj,ins,mag_nn,itp_lin,:ekf;P0=P0,Qd=Qd,R=R,
                                      core=true,run_crlb=false)).lat,fo.lon)
        catch e; @warn("ekf",e); Inf end
        d_mpf = try
            if any(isnan,mag_nn)
                NaN
            else
                fr = mpf_logw(ins,mag_nn,itp_lin;P0=P0,Qd=Qd,R=R,num_part=NP,
                              thresh=THR,roughen_m=RGH,core=true)
                fr.c ? drms_ll(traj,(fo=MagNav.eval_filt(traj,ins,fr)).lat,fo.lon) : Inf
            end
        catch e; @warn("mpf_logw",e); Inf end

        push!(results,(fl,line,tag,round(crms,digits=1),round(d_ekf,digits=1),round(d_mpf,digits=1)))
        println("  $tag  comp-RMSE=$(round(crms,digits=1)) nT   EKF+offNN=$(round(d_ekf,digits=1)) m   MPF+offNN(fixed)=$(round(d_mpf,digits=1)) m")
    end
end

CSV.write(joinpath(@__DIR__,"mpf_offnn_results.csv"), results)
println("\n=== frozen offline-NN compensation + repaired native MPF ===")
show(results, allrows=true, allcols=true); println()
println("wrote mpf_offnn_results.csv")
