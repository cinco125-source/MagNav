##* Cold-start error trace for the graphical abstract (Fig. 1c).
#
# The committed track_data.csv is the navigation-only comparison on the COMPENSATED
# Mag 1 (batch FGO vs EKF vs INS) and is NOT the cold-start compensation story. This
# script produces the honest cold-start trace: on the UNCOMPENSATED cabin
# magnetometers (Mag 4, Mag 5) of line 1003.02, the online-TL EKF and the fixed-lag
# window FGO both start from beta = 0 with no calibration, and we dump their
# per-epoch horizontal error [m] vs time so make_figures.py can plot a real curve
# instead of a cartoon. Downsampled to ~300 rows; written to research/coldstart_trace.csv.
#
# Usage: julia --project=. research/coldstart_trace.jl

using MagNav
using CSV, DataFrames
using LinearAlgebra, Statistics

MEAS_VAR = 12.0^2
FOGM_SIG = 3.0
FOGM_TAU = 180.0
TERMS    = [:permanent,:induced,:eddy,:bias]

flight   = :Flt1003
line     = 1003.02
map_name = :Eastern_395

df_dir    = joinpath(@__DIR__,"..","examples","dataframes")
df_flight = DataFrame(CSV.File(joinpath(df_dir,"df_flight.csv")))
df_flight[!,:flight]   = Symbol.(df_flight[!,:flight])
df_flight[!,:xyz_type] = Symbol.(df_flight[!,:xyz_type])
df_flight[!,:xyz_file] = String.(df_flight[!,:xyz_file])
for (i,fl) in enumerate(df_flight.flight)
    fl == flight || continue
    df_flight.xyz_file[i] = MagNav.sgl_2020_train(fl)
end
df_map = DataFrame(CSV.File(joinpath(df_dir,"df_map.csv")))
df_map[!,:map_name] = Symbol.(df_map[!,:map_name])
df_map[!,:map_file] = String.(df_map[!,:map_file])
for (i,mn) in enumerate(df_map.map_name)
    mn == map_name || continue
    df_map.map_file[i] = MagNav.ottawa_area_maps(mn)
end
df_nav = DataFrame(CSV.File(joinpath(df_dir,"df_nav.csv")))
df_nav[!,:flight]   = Symbol.(df_nav[!,:flight])
df_nav[!,:map_name] = Symbol.(df_nav[!,:map_name])

xyz  = get_XYZ(flight,df_flight;silent=true)
ind  = get_ind(xyz,line,df_nav)
mapS = get_map(map_name,df_map)
traj = get_traj(xyz,ind)
ins  = get_ins(xyz,ind;N_zero_ll=1)
(_,itp) = get_map_val(mapS,traj;return_itp=true)
flux = xyz.flux_d(ind)
nTL  = size(create_TL_A(flux;terms=TERMS),2)

(P0,Qd,R) = create_model(traj.dt,traj.lat[1];init_pos_sigma=0.1,init_alt_sigma=1.0,
    init_vel_sigma=1.0,meas_var=MEAS_VAR,fogm_sigma=FOGM_SIG,fogm_tau=FOGM_TAU,
    vec_states=false,TL_sigma=fill(1.0,nTL),P0_TL=Matrix(Diagonal(fill(1.0,nTL))))

herr(lat,lon) = sqrt.(dlat2dn.(lat .- traj.lat, traj.lat).^2 .+
                      dlon2de.(lon .- traj.lon, traj.lat).^2)
tt = (traj.tt .- traj.tt[1]) ./ 60                      # [min]
e_ins = herr(ins.lat, ins.lon)

function trace(magsym)
    mag = getfield(xyz,magsym)[ind]
    e_ekf = fill(NaN, length(tt)); e_fgo = fill(NaN, length(tt))
    try
        fo = run_filt(traj,ins,mag,itp,:ekf_online;P0=P0,Qd=Qd,R=R,
                      flux=flux,x0_TL=zeros(nTL),terms=TERMS,core=true,run_crlb=false)
        e_ekf = herr(fo.lat,fo.lon)
    catch e; @warn("ekf_online failed $magsym",e) end
    try
        fr = fgo_online(ins,mag,flux,itp,zeros(nTL),P0,Qd,R;terms=TERMS,core=true,
                        win=300.0,overlap=90.0,robust=:huber)
        fo = MagNav.eval_filt(traj,ins,fr)
        e_fgo = herr(fo.lat,fo.lon)
    catch e; @warn("fgo_online failed $magsym",e) end
    drms(e) = sqrt(mean(filter(isfinite,e).^2))
    println("  $magsym  EKF-online DRMS=$(round(drms(e_ekf),digits=1))  ",
            "FGO DRMS=$(round(drms(e_fgo),digits=1)) m")
    (e_ekf, e_fgo)
end

println("cold-start trace, Flt1003 1003.02:")
(e_ekf4, e_fgo4) = trace(:mag_4_uc)
(e_ekf5, e_fgo5) = trace(:mag_5_uc)

step = max(1, div(length(tt),300))
d    = 1:step:length(tt)
out  = DataFrame(tmin=round.(tt[d],digits=3),
                 e_ins  =round.(e_ins[d],  digits=2),
                 e_ekf_m4=round.(e_ekf4[d],digits=2), e_fgo_m4=round.(e_fgo4[d],digits=2),
                 e_ekf_m5=round.(e_ekf5[d],digits=2), e_fgo_m5=round.(e_fgo5[d],digits=2))
CSV.write(joinpath(@__DIR__,"coldstart_trace.csv"), out)
println("wrote coldstart_trace.csv  ($(nrow(out)) rows)")
