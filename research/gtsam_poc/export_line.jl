##* GTSAM PoC — export one line's FGO ingredients to HDF5 for a Python gtsam
# IncrementalFixedLagSmoother reproduction.
#
# Exports a segment of line 1007.06 (cold-start, uncompensated cabin Mag 5): the
# error-state transition tensor Phi[nx,nx,N-1], the Tolles-Lawson rows A[N,nTL], the
# scalar measurement, the INS reference position (for the error-state offset), the
# true position (for DRMS), the noise blocks (P0, Qd, R), and a (map + IGRF core)
# grid sampled at the mean altitude over the trajectory so Python can interpolate
# h(pos) for relinearization. Also runs the Julia fixed-lag FGO on the same segment
# to record the reference DRMS the Python side must reproduce.
#
# Segment length keeps the dense Phi tensor small enough to ship through CI.
#
# Usage: julia --project=. research/gtsam_poc/export_line.jl

using MagNav
using CSV, DataFrames, HDF5
using LinearAlgebra, Statistics
using Random: seed!
seed!(33)

const SEG_MIN = 10.0                       # segment length [min] (Phi size ~ nx^2 * N)
const MAGSYM  = :mag_5_uc                   # cleaner leg for the first PoC
const TERMS   = [:permanent,:induced,:eddy,:bias]
const MEAS_VAR, FOGM_SIG, FOGM_TAU = 12.0^2, 3.0, 180.0
const WIN, OVERLAP = 300.0, 90.0
fl, line = :Flt1007, 1007.06

df_dir    = joinpath(@__DIR__,"..","..","examples","dataframes")
df_flight = DataFrame(CSV.File(joinpath(df_dir,"df_flight.csv")))
df_flight[!,:flight]   = Symbol.(df_flight[!,:flight])
df_flight[!,:xyz_type] = Symbol.(df_flight[!,:xyz_type])
df_flight[!,:xyz_file] = String.(df_flight[!,:xyz_file])
for (i,f) in enumerate(df_flight.flight)
    f == fl && (df_flight.xyz_file[i] = MagNav.sgl_2020_train(f))
end
df_map = DataFrame(CSV.File(joinpath(df_dir,"df_map.csv")))
df_map[!,:map_name] = Symbol.(df_map[!,:map_name])
df_map[!,:map_file] = String.(df_map[!,:map_file])
for (i,m) in enumerate(df_map.map_name)
    df_map.map_file[i] = MagNav.ottawa_area_maps(m)
end
df_nav = DataFrame(CSV.File(joinpath(df_dir,"df_nav.csv")))
df_nav[!,:flight]   = Symbol.(df_nav[!,:flight])
df_nav[!,:map_name] = Symbol.(df_nav[!,:map_name])

xyz   = get_XYZ(fl,df_flight;silent=true)
ind   = get_ind(xyz,line,df_nav)
mname = df_nav[(df_nav.flight.==fl).&(df_nav.line.==line),:map_name][1]
mapS  = get_map(mname,df_map)
traj  = get_traj(xyz,ind)
ins   = get_ins(xyz,ind;N_zero_ll=1)
(_,itp) = get_map_val(mapS,traj;return_itp=true)
flux  = xyz.flux_d(ind)

Nseg = min(traj.N, round(Int, SEG_MIN*60/traj.dt))
S    = 1:Nseg
dt   = traj.dt
date = MagNav.get_years(2020,185)

A   = create_TL_A(flux;terms=TERMS)          # [Nfull, nTL]
nTL = size(A,2)
(P0,Qd,R) = create_model(dt,traj.lat[1];init_pos_sigma=0.1,init_alt_sigma=1.0,
    init_vel_sigma=1.0,meas_var=MEAS_VAR,fogm_sigma=FOGM_SIG,fogm_tau=FOGM_TAU,
    vec_states=false,TL_sigma=fill(1.0,nTL),P0_TL=Matrix(Diagonal(fill(1.0,nTL))))
nx = size(P0,1)

# error-state transition Phi[:,:,t] over the segment (t = 1..Nseg-1)
Phi = zeros(Float32,nx,nx,Nseg-1)
for t = 1:Nseg-1
    Phi[:,:,t] = MagNav.get_Phi(nx,ins.lat[t],ins.vn[t],ins.ve[t],ins.vd[t],
                         ins.fn[t],ins.fe[t],ins.fd[t],ins.Cnb[:,:,t],
                         3600.0,3600.0,3600.0,FOGM_TAU,dt)
end

# (map + IGRF core) grid at the mean segment altitude, over a padded bbox, so the
# Python side can evaluate h(pos) and its gradient for relinearization.
malt = mean(ins.alt[S])
pad  = 0.02*pi/180                            # ~2 km padding [rad]
Ng   = 200
glat = collect(range(minimum(traj.lat[S])-pad, maximum(traj.lat[S])+pad, length=Ng))
glon = collect(range(minimum(traj.lon[S])-pad, maximum(traj.lon[S])+pad, length=Ng))
gh   = zeros(Float64,Ng,Ng)                   # gh[i,j] = map(glat_i,glon_j,malt)+|IGRF|
for i = 1:Ng, j = 1:Ng
    m = itp(glat[i],glon[j],malt)
    c = norm(MagNav.igrf(date,malt,glat[i],glon[j],Val(:geodetic)))
    gh[i,j] = m + c
end

meas = getfield(xyz,MAGSYM)[ind][S]

# reference: Julia fixed-lag FGO (marginalized handoff) DRMS on the same segment
function drms(tt,tlat,tlon,lat,lon;warm=600.0)
    m  = (tt .- tt[1]) .>= warm
    dn = dlat2dn.(lat[m] .- tlat[m], tlat[m]); de = dlon2de.(lon[m] .- tlon[m], tlat[m])
    sqrt(mean(dn.^2 .+ de.^2))
end
# segment mask = first Nseg TRUE samples of the full-flight line mask `ind`
idx     = findall(ind)
segmask = falses(length(ind)); segmask[idx[1:Nseg]] .= true
traj_s = get_traj(xyz,segmask); ins_s = get_ins(xyz,segmask;N_zero_ll=1)
fr  = fgo_online(ins_s,meas,xyz.flux_d(segmask),itp,zeros(nTL),P0,Qd,R;terms=TERMS,
                 core=true,win=WIN,overlap=OVERLAP,robust=:huber)
fo  = MagNav.eval_filt(traj_s,ins_s,fr)
ref_drms = drms(traj_s.tt,traj_s.lat,traj_s.lon,fo.lat,fo.lon;warm=60.0)
println("Julia fixed-lag FGO reference DRMS ($(MAGSYM), $(round(Nseg*dt/60,digits=1)) min) = $(round(ref_drms,digits=2)) m")

out = joinpath(@__DIR__,"line_1007_06.h5")
h5open(out,"w") do f
    f["N"]        = Nseg
    f["dt"]       = dt
    f["nx"]       = nx
    f["nTL"]      = nTL
    f["Phi"]      = Phi                        # [nx,nx,N-1]
    f["A"]        = Matrix{Float64}(A[S,:])    # [N,nTL]
    f["meas"]     = collect(Float64,meas)      # [N]
    f["ins_lat"]  = collect(Float64,ins.lat[S])
    f["ins_lon"]  = collect(Float64,ins.lon[S])
    f["ins_alt"]  = collect(Float64,ins.alt[S])
    f["true_lat"] = collect(Float64,traj.lat[S])
    f["true_lon"] = collect(Float64,traj.lon[S])
    f["P0"]       = Matrix{Float64}(P0)
    f["Qd"]       = Matrix{Float64}(Qd)
    f["R"]        = Float64(mean(R))
    f["glat"]     = glat
    f["glon"]     = glon
    f["gh"]       = gh
    f["malt"]     = malt
    f["warm"]     = 60.0
    f["win"]      = WIN
    f["overlap"]  = OVERLAP
    f["ref_drms"] = ref_drms
end
println("wrote $out  (N=$Nseg, nx=$nx, nTL=$nTL, grid=$(Ng)x$(Ng))")
