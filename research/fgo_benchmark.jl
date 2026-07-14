##* FGO vs EKF benchmark on real SGL 2020 flight data (Flt1003, Eastern Ontario)
# Quantitative comparison of navigation accuracy (DRMS) and runtime for:
#   1. EKF                  - causal extended Kalman filter (baseline)
#   2. FGO (RTS)            - batch MAP smoother, iterated RTS solver
#   3. FGO (RTS,huber)      - with robust (Huber) measurement kernel
#   4. FGO (GN/QR)          - global sparse square-root Gauss-Newton solver
#                             (run on a shorter segment, see below)
#   5. EKF online           - EKF with online Tolles-Lawson estimation
#   6. FGO online           - batch FGO with Tolles-Lawson factors
#
# Data: SGL 2020 training data (sgl_2020_train artifact), flight Flt1003,
# navigation-capable line 1003.02 over the Eastern_395 map (ottawa_area_maps
# artifact). Artifacts are downloaded automatically (lazy artifacts).
#
# Usage: julia --project=. research/fgo_benchmark.jl

using MagNav
using CSV, DataFrames
using LinearAlgebra, Statistics
using Random: seed!

seed!(33) # for reproducibility

##* dataframes for SGL flight data & maps (as in examples/dataframes_setup.jl)
df_dir    = joinpath(@__DIR__,"..","examples","dataframes")
df_flight = DataFrame(CSV.File(joinpath(df_dir,"df_flight.csv")))
df_flight[!,:flight]   = Symbol.(df_flight[!,:flight])
df_flight[!,:xyz_type] = Symbol.(df_flight[!,:xyz_type])
df_flight[!,:xyz_file] = String.(df_flight[!,:xyz_file])
for (i,flight) in enumerate(df_flight.flight)
    flight in (:Flt1002,:Flt1003) || continue # only resolve needed artifacts
    df_flight.xyz_type[i] == :XYZ20 && (df_flight.xyz_file[i] = MagNav.sgl_2020_train(flight))
    df_flight.xyz_type[i] == :XYZ21 && (df_flight.xyz_file[i] = MagNav.sgl_2021_train(flight))
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

##* flight data & map
flight   = :Flt1003
line     = 1003.02       # full nav-capable line
map_name = :Eastern_395

@info("loading flight data $flight")
xyz  = get_XYZ(flight,df_flight;silent=true)
ind  = get_ind(xyz,line,df_nav)
mapS = get_map(map_name,df_map)

traj = get_traj(xyz,ind)             # trajectory (GPS) struct
ins  = get_ins( xyz,ind;N_zero_ll=1) # INS struct, zeroed to first traj point
(map_val,itp_mapS) = get_map_val(mapS,traj;return_itp=true)

mag_use = xyz.mag_1_c[ind] # stinger magnetometer, compensated
println("line $line: N = $(traj.N) samples, dt = $(traj.dt) s, ",
        "~$(round(traj.N*traj.dt/60,digits=1)) min")
println("map-mag std: $(round(std(map_val+(xyz.diurnal+xyz.igrf)[ind]-mag_use),digits=2)) nT")

##* filter model (per examples/pluto_sgl.jl)
(P0,Qd,R) = create_model(traj.dt,traj.lat[1];
                         init_pos_sigma = 0.1,
                         init_alt_sigma = 1.0,
                         init_vel_sigma = 1.0,
                         meas_var       = 5^2,
                         fogm_sigma     = 3,
                         fogm_tau       = 180);

##* horizontal position DRMS [m] from extracted filter output
function drms_out(filt_out, traj)
    dn = dlat2dn.(filt_out.lat .- traj.lat, traj.lat)
    de = dlon2de.(filt_out.lon .- traj.lon, traj.lat)
    sqrt(mean(dn.^2 .+ de.^2))
end

ins_drms = sqrt(mean(dlat2dn.(ins.lat .- traj.lat, traj.lat).^2 .+
                     dlon2de.(ins.lon .- traj.lon, traj.lat).^2))

results = DataFrame(method=String[],segment=String[],drms=Float64[],time=Float64[])

function run_and_log!(results, method, segment, traj, ins, meas, itp, filt_type;
                      kwargs...)
    try
        t = @elapsed filt_out = run_filt(traj,ins,meas,itp,filt_type;
                                         kwargs...,core=true,run_crlb=false)
        d = drms_out(filt_out,traj)
        push!(results,(method,segment,d,t))
        println(rpad(method,22)," DRMS = ",rpad(round(d,digits=1),8),
                " m,  time = ",round(t,digits=1)," s")
        return filt_out
    catch e
        @warn("$method failed",e)
        push!(results,(method,segment,NaN,NaN))
        return nothing
    end
end

##* full line: EKF vs FGO (RTS) vs FGO (RTS,huber)
println("\n=== full line $line ($(traj.N) samples) ===")
println(rpad("INS (no aiding)",22)," DRMS = ",round(ins_drms,digits=1)," m")
run_and_log!(results,"EKF","full",traj,ins,mag_use,itp_mapS,:ekf;
             P0,Qd,R)
run_and_log!(results,"FGO (RTS)","full",traj,ins,mag_use,itp_mapS,:fgo;
             P0,Qd,R)
run_and_log!(results,"FGO (RTS,huber)","full",traj,ins,mag_use,itp_mapS,:fgo;
             P0,Qd,R,robust=:huber)

##* segment (first 10 min): EKF vs FGO (RTS) vs FGO (GN/QR sparse solver)
#* the global sparse QR solver memory scales with N, so a segment is used
tt0     = xyz.traj.tt[ind][1]
ind_seg = get_ind(xyz;tt_lim=[tt0,tt0+600.0])
traj_s  = get_traj(xyz,ind_seg)
ins_s   = get_ins( xyz,ind_seg;N_zero_ll=1)
mag_s   = xyz.mag_1_c[ind_seg]
itp_s   = get_map_val(mapS,traj_s;return_itp=true)[2]

println("\n=== segment ($(traj_s.N) samples, 10 min) ===")
run_and_log!(results,"EKF","segment",traj_s,ins_s,mag_s,itp_s,:ekf;
             P0,Qd,R)
run_and_log!(results,"FGO (RTS)","segment",traj_s,ins_s,mag_s,itp_s,:fgo;
             P0,Qd,R)
run_and_log!(results,"FGO (GN/QR)","segment",traj_s,ins_s,mag_s,itp_s,:fgo;
             P0,Qd,R,solver=:gn)
run_and_log!(results,"FGO (GN/QR,huber)","segment",traj_s,ins_s,mag_s,itp_s,:fgo;
             P0,Qd,R,solver=:gn,robust=:huber)

##* online Tolles-Lawson: EKF online vs FGO online (batch compensation factors)
#* uncompensated cabin magnetometer 4 + fluxgate D, TL initialized from the
#* Flt1002 calibration box (line 1002.02), per standard SGL 2020 usage
println("\n=== online Tolles-Lawson (mag 4 uncompensated) ===")
@info("loading flight data Flt1002 for TL calibration")
xyz_cal = get_XYZ(:Flt1002,df_flight;silent=true)
TL_i    = findfirst((df_cal.flight .== :Flt1002) .& (df_cal.line .== 1002.02))
TL_ind  = get_ind(xyz_cal;tt_lim=[df_cal.t_start[TL_i],df_cal.t_end[TL_i]])
(x0_TL,P0_TL,TL_sigma) = ekf_online_setup(xyz_cal.flux_d,xyz_cal.mag_4_uc,TL_ind)
(P0_o,Qd_o,R_o) = create_model(traj.dt,traj.lat[1];
                               init_pos_sigma = 0.1,
                               init_alt_sigma = 1.0,
                               init_vel_sigma = 1.0,
                               meas_var       = 5^2,
                               fogm_sigma     = 3,
                               fogm_tau       = 180,
                               vec_states     = false,
                               TL_sigma       = TL_sigma,
                               P0_TL          = P0_TL);

#* larger position excursions (from the noisier cabin magnetometer) can
#* slightly exceed the map borders near the north end of the line, so the
#* interpolation is clamped to the map borders (edge value used outside)
(lat_lo,lat_hi) = extrema(mapS.yy)
(lon_lo,lon_hi) = extrema(mapS.xx)
pad = 1e-7 # [rad] stay strictly inside the borders for cubic interpolation
itp_clamp = (lat,lon,alt) -> itp_mapS(clamp(lat,lat_lo+pad,lat_hi-pad),
                                      clamp(lon,lon_lo+pad,lon_hi-pad),alt)

flux_use = xyz.flux_d(ind)
mag_uc   = xyz.mag_4_uc[ind]
run_and_log!(results,"EKF online","full",traj,ins,mag_uc,itp_clamp,:ekf_online;
             P0=P0_o,Qd=Qd_o,R=R_o,flux=flux_use,x0_TL=x0_TL)
run_and_log!(results,"FGO online","full",traj,ins,mag_uc,itp_clamp,:fgo_online;
             P0=P0_o,Qd=Qd_o,R=R_o,flux=flux_use,x0_TL=x0_TL)
run_and_log!(results,"FGO online (huber)","full",traj,ins,mag_uc,itp_clamp,:fgo_online;
             P0=P0_o,Qd=Qd_o,R=R_o,flux=flux_use,x0_TL=x0_TL,robust=:huber)

##* results summary
println("\n=== results summary ===")
println("INS (no aiding) DRMS: $(round(ins_drms,digits=1)) m")
show(results;allrows=true,allcols=true)
println()
out_csv = joinpath(@__DIR__,"fgo_benchmark_results.csv")
CSV.write(out_csv,results)
println("\nresults written to $out_csv")
