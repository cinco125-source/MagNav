##* Geographic tracks + position-error plots for FGO vs EKF on real Flt1003 data
# Produces two figures for the SGL 2020 Flt1003 nav line 1003.02 (Eastern_395):
#   1. fgo_map_track.png  — the magnetic anomaly map with the flight line drawn
#                           on it (geographic context: the field the nav uses)
#   2. fgo_pos_error.png  — horizontal position error [m] vs time for INS, EKF,
#                           and FGO (at 40 km map scale the tracks overlap, so
#                           this is where the performance is actually visible)
#   3. fgo_zoom_track.png — a short zoomed window of the estimated ground tracks
#
# Usage: julia --project=. research/fgo_tracks.jl

using MagNav
using CSV, DataFrames
using LinearAlgebra, Statistics
using Plots
using Random: seed!
gr()
seed!(33)

df_dir    = joinpath(@__DIR__,"..","examples","dataframes")
df_flight = DataFrame(CSV.File(joinpath(df_dir,"df_flight.csv")))
df_flight[!,:flight]   = Symbol.(df_flight[!,:flight])
df_flight[!,:xyz_type] = Symbol.(df_flight[!,:xyz_type])
df_flight[!,:xyz_file] = String.(df_flight[!,:xyz_file])
for (i,flight) in enumerate(df_flight.flight)
    flight == :Flt1003 || continue
    df_flight.xyz_file[i] = MagNav.sgl_2020_train(flight)
end
df_map = DataFrame(CSV.File(joinpath(df_dir,"df_map.csv")))
df_map[!,:map_name] = Symbol.(df_map[!,:map_name])
df_map[!,:map_file] = String.(df_map[!,:map_file])
for (i,map_name) in enumerate(df_map.map_name)
    map_name == :Eastern_395 || continue
    df_map.map_file[i] = MagNav.ottawa_area_maps(map_name)
end
df_nav = DataFrame(CSV.File(joinpath(df_dir,"df_nav.csv")))
df_nav[!,:flight]   = Symbol.(df_nav[!,:flight])
df_nav[!,:map_name] = Symbol.(df_nav[!,:map_name])

flight   = :Flt1003
line     = 1003.02
map_name = :Eastern_395

@info("loading flight data $flight")
xyz  = get_XYZ(flight,df_flight;silent=true)
ind  = get_ind(xyz,line,df_nav)
mapS = get_map(map_name,df_map)
traj = get_traj(xyz,ind)
ins  = get_ins( xyz,ind;N_zero_ll=1)
itp_mapS = map_interpolate(mapS)
mag_use  = xyz.mag_1_c[ind]

(P0,Qd,R) = create_model(traj.dt,traj.lat[1];
                         init_pos_sigma=0.1,init_alt_sigma=1.0,init_vel_sigma=1.0,
                         meas_var=5^2,fogm_sigma=3,fogm_tau=180)

ekf_out = run_filt(traj,ins,mag_use,itp_mapS,:ekf;P0,Qd,R,core=true,run_crlb=false)
fgo_out = run_filt(traj,ins,mag_use,itp_mapS,:fgo;P0,Qd,R,core=true,run_crlb=false)

##* 1) magnetic anomaly map + flight line (geographic context)
p_map = plot_map(mapS;dpi=200)
plot_path!(p_map,traj;path_color=:black,lab="flight line 1003.02",show_plot=false)
savefig(p_map,joinpath(@__DIR__,"fgo_map_track.png"))

##* 2) horizontal position error [m] vs time  (the performance view)
herr(lat,lon) = sqrt.(dlat2dn.(lat .- traj.lat, traj.lat).^2 .+
                      dlon2de.(lon .- traj.lon, traj.lat).^2)
tt = (traj.tt .- traj.tt[1]) ./ 60 # [min]
e_ins = herr(ins.lat,ins.lon)
e_ekf = herr(ekf_out.lat,ekf_out.lon)
e_fgo = herr(fgo_out.lat,fgo_out.lon)

p_err = plot(tt,e_ins;lab="INS  (DRMS $(round(sqrt(mean(e_ins.^2)),digits=0)) m)",
             color=:gray,lw=2,dpi=200,xlab="time [min]",ylab="horizontal error [m]",
             title="Flt1003 line 1003.02 — position error",legend=:topleft,
             ylim=(0,quantile(e_ins,0.99)*1.05))
plot!(p_err,tt,e_ekf;lab="EKF  (DRMS $(round(sqrt(mean(e_ekf.^2)),digits=1)) m)",
      color=RGB(0.16,0.47,0.84),lw=2)
plot!(p_err,tt,e_fgo;lab="FGO  (DRMS $(round(sqrt(mean(e_fgo.^2)),digits=1)) m)",
      color=RGB(0.10,0.69,0.48),lw=2.5)
savefig(p_err,joinpath(@__DIR__,"fgo_pos_error.png"))

##* 3) zoomed ground tracks over a 3-min window (differences become visible)
i0 = findfirst(tt .>= tt[end]/2)          # mid-flight
i1 = findfirst(tt .>= tt[i0] + 3)         # +3 min
i1 = i1 === nothing ? length(tt) : i1
w  = i0:i1
p_zoom = plot(rad2deg.(traj.lon[w]),rad2deg.(traj.lat[w]);lab="truth",color=:black,
              lw=3,dpi=200,xlab="longitude [deg]",ylab="latitude [deg]",
              title="zoomed ground track (3 min)",legend=:best,aspect_ratio=:equal)
plot!(p_zoom,rad2deg.(ins.lon[w]),rad2deg.(ins.lat[w]);lab="INS",color=:gray,lw=2,ls=:dash)
plot!(p_zoom,rad2deg.(ekf_out.lon[w]),rad2deg.(ekf_out.lat[w]);
      lab="EKF",color=RGB(0.16,0.47,0.84),lw=2)
plot!(p_zoom,rad2deg.(fgo_out.lon[w]),rad2deg.(fgo_out.lat[w]);
      lab="FGO",color=RGB(0.10,0.69,0.48),lw=2.5)
savefig(p_zoom,joinpath(@__DIR__,"fgo_zoom_track.png"))

println("wrote fgo_map_track.png, fgo_pos_error.png, fgo_zoom_track.png")
println("DRMS  INS=$(round(sqrt(mean(e_ins.^2)),digits=1))  ",
        "EKF=$(round(sqrt(mean(e_ekf.^2)),digits=1))  ",
        "FGO=$(round(sqrt(mean(e_fgo.^2)),digits=1)) m")

##* also print downsampled tracks + map anomaly grid as CSV (for offline plots)
step = max(1, div(length(tt),300))
d = 1:step:length(tt)
println("###TRACKCSV_START")
println("tmin,tlat,tlon,ilat,ilon,elat,elon,flat,flon,eins,eekf,efgo")
for i in d
    println(join(round.([tt[i],
        rad2deg(traj.lat[i]),rad2deg(traj.lon[i]),
        rad2deg(ins.lat[i]),rad2deg(ins.lon[i]),
        rad2deg(ekf_out.lat[i]),rad2deg(ekf_out.lon[i]),
        rad2deg(fgo_out.lat[i]),rad2deg(fgo_out.lon[i]),
        e_ins[i],e_ekf[i],e_fgo[i]],digits=7),","))
end
println("###TRACKCSV_END")

# coarse map anomaly grid over the flight bbox for a background heatmap
latlo,lathi = extrema(traj.lat); lonlo,lonhi = extrema(traj.lon)
padlat = (lathi-latlo)*0.08; padlon = (lonhi-lonlo)*0.08
glat = range(latlo-padlat,lathi+padlat,length=60)
glon = range(lonlo-padlon,lonhi+padlon,length=60)
println("###MAPCSV_START")
println("nlat,nlon,latlo,lathi,lonlo,lonhi")
println(join([60,60,rad2deg(first(glat)),rad2deg(last(glat)),
              rad2deg(first(glon)),rad2deg(last(glon))],","))
for la in glat
    row = [ itp_mapS(la,lo,mapS.alt) for lo in glon ]
    println(join(round.(row,digits=2),","))
end
println("###MAPCSV_END")
