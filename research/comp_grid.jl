##* The estimator x compensation grid: filling the missing cells.
#
# Paper-facing comparison the breadth study implies but never tabulates in full:
#
#   compensation:   none | offline NN (Gnadt :m1) | online TL | online TL+NN | joint FGO
#   estimator:      EKF  | MPF                    | (window smoother = ours)
#
# The online columns exist (fgo_breadth.jl, mpf_nn.jl). This script fills the
# offline-NN cells on the counted lines x Mag 4/5:
#   EKF + offline NN and MPF + offline NN -- Gnadt-style model :m1 trained on
#   OTHER flights (Flt1002+Flt1006), y_type=:d (cabin interference vs compensated
#   stinger), TL-A + current/voltage features, then frozen and applied
#   cross-flight. (No-compensation anchors were dropped: they diverge trivially
#   and add no information.)
#
# Outputs research/comp_grid_results.csv. Usage:
#   julia --project=. research/comp_grid.jl

using MagNav
using CSV, DataFrames
using LinearAlgebra, Statistics
using Random: seed!
seed!(33)

MEAS_VAR = 12.0^2
FOGM_SIG = 3.0
FOGM_TAU = 180.0
NUM_PART = 300
LINES = [(:Flt1003,1003.02), (:Flt1003,1003.08),
         (:Flt1007,1007.02), (:Flt1007,1007.06)]
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

##* train one offline :m1 model per cabin magnetometer --------------------------
models = Dict{Symbol,Any}()
for magsym in (:mag_4_uc,:mag_5_uc)
    println("training offline :m1 for $magsym on $(TRAIN_FLIGHTS) ",
            "($(length(train_lines)) lines)")
    cp = MagNav.NNCompParams(features_setup=FEATURES, model_type=:m1, y_type=:d,
                             use_mag=magsym, use_vec=:flux_d,
                             epoch_adam=20, hidden=[8], batchsize=2048)
    (cp_out,_,_,_,_) = comp_train(cp, train_lines, df_all, df_flight, df_map;
                                  silent=true)
    models[magsym] = cp_out
end

##* fill the four grid cells on the counted lines -------------------------------
results = DataFrame(flight=Symbol[],line=Float64[],mag=String[],cell=String[],
                    comp_rmse_nT=Float64[],DRMS=Float64[])

for (fl,line) in LINES
    xyz   = getxyz(fl)
    ind   = get_ind(xyz,line,df_nav)
    mname = df_nav[(df_nav.flight.==fl).&(df_nav.line.==line),:map_name][1]
    mapS  = get_map(mname,df_map)
    traj  = get_traj(xyz,ind)
    ins   = get_ins(xyz,ind;N_zero_ll=1)
    (_,itp) = get_map_val(mapS,traj;return_itp=true)
    ref   = xyz.mag_1_c[ind]
    flux  = xyz.flux_d(ind)
    TERMS = [:permanent,:induced,:eddy,:bias]
    nTL   = size(create_TL_A(flux;terms=TERMS),2)
    (P0,Qd,R) = create_model(traj.dt,traj.lat[1];init_pos_sigma=0.1,
        init_alt_sigma=1.0,init_vel_sigma=1.0,meas_var=MEAS_VAR,
        fogm_sigma=FOGM_SIG,fogm_tau=FOGM_TAU)
    (P0t,Qdt,Rt) = create_model(traj.dt,traj.lat[1];init_pos_sigma=0.1,
        init_alt_sigma=1.0,init_vel_sigma=1.0,meas_var=MEAS_VAR,
        fogm_sigma=FOGM_SIG,fogm_tau=FOGM_TAU,vec_states=false,
        TL_sigma=fill(1.0,nTL),P0_TL=Matrix(Diagonal(fill(1.0,nTL))))
    println("\n$fl $line ($mname)")
    for magsym in (:mag_4_uc,:mag_5_uc)
        mag = getfield(xyz,magsym)[ind]
        tag = replace(String(magsym),"mag_"=>"Mag ","_uc"=>"")

        # offline-NN compensated signal for this line (frozen cross-flight model)
        mag_nn = fill(NaN,length(mag))
        try
            (_,y_hat,_,_) = comp_test(models[magsym], xyz, ind; silent=true)
            mag_nn = mag .- y_hat
        catch e; @warn("comp_test failed $fl $line $tag",e) end

        for (cell,z) in (("EKF offNN",mag_nn),("MPF offNN",mag_nn),
                         ("EKFonTL offNN",mag_nn))
            crms = any(isnan,z) ? NaN : sqrt(mean((z .- ref).^2))
            d = Inf
            try
                if any(isnan,z)
                    d = NaN
                elseif cell == "EKFonTL offNN"
                    # user-proposed stack: offline NN strips the nonlinear bulk,
                    # online TL tracks the time-varying residual during navigation
                    fo = run_filt(traj,ins,z,itp,:ekf_online;P0=P0t,Qd=Qdt,R=Rt,
                                  flux=flux,x0_TL=zeros(nTL),terms=TERMS,
                                  core=true,run_crlb=false)
                    d  = drms_ll(traj,fo.lat,fo.lon)
                elseif startswith(cell,"EKF")
                    fo = run_filt(traj,ins,z,itp,:ekf;P0=P0,Qd=Qd,R=R,
                                  core=true,run_crlb=false)
                    d  = drms_ll(traj,fo.lat,fo.lon)
                else
                    fo = run_filt(traj,ins,z,itp,:mpf;P0=P0,Qd=Qd,R=R,
                                  core=true,run_crlb=false,num_part=NUM_PART)
                    d  = drms_ll(traj,fo.lat,fo.lon)
                end
            catch e; @warn("$cell failed $fl $line $tag",e) end
            push!(results,(fl,line,tag,cell,round(crms,digits=1),round(d,digits=1)))
            println("  $tag $(rpad(cell,10)) comp-RMSE=$(round(crms,digits=1)) nT",
                    "  DRMS=$(round(d,digits=1)) m")
        end
    end
end

println("\n=== estimator x compensation grid (missing cells) ===")
show(results;allrows=true,allcols=true); println()
CSV.write(joinpath(@__DIR__,"comp_grid_results.csv"),results)
println("wrote comp_grid_results.csv")
