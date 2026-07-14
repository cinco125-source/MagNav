##* Sensor-error factor ablation for FGO MagNav (factorial "각 경우의 수" study)
# A simulated flight over a HIGH-RESOLUTION survey map, with heading variation,
# is corrupted with KNOWN, physically-motivated scalar-magnetometer errors of an
# optically-pumped / quantum sensor (Oelsner et al. 2022):
#   - heading error   Δ_head = Σ_{n∈{1,2,4}} c_n cos(n ψ)  (OPM light shift +
#                     nonlinear Zeeman; Hager et al. 2026, Wang et al. 2020 GRSL)
#   - hard-iron bias  Δ_bias = m · û_body(t)          (fluxgate vector bias)
#   - linear drift    Δ_drift = d · t                 (electronics drift)
#   - dead zones      noise amplified by 1/max(|cos ψ|,ε) (OPM equatorial dead
#                     zone at ψ = 90°; Hager et al. 2026 eq. 21)
# The batch FGO is run with each combination of sensor-error factors and we
# report navigation error (DRMS) plus how well each factor RECOVERS its truth.
#
# The sensor optical axis is body-x (nose): as the aircraft yaws, θ sweeps a wide
# range, so the heading error / dead zones are observable (a body-down axis would
# leave θ ≈ const at the field inclination — unobservable, absorbed by a bias).
#
# Usage: julia --project=. research/fgo_sensor_ablation.jl

using MagNav
using CSV, DataFrames
using LinearAlgebra, Statistics
using Random: seed!

seed!(2)

##* high-resolution survey map (Eastern_395); fall back to coarse NAMAD
try
    df_map = DataFrame(CSV.File(joinpath(@__DIR__,"..","examples","dataframes","df_map.csv")))
    df_map[!,:map_name] = Symbol.(df_map[!,:map_name])
    df_map[!,:map_file] = String.(df_map[!,:map_file])
    for i in eachindex(df_map.map_name)
        df_map.map_name[i] == :Eastern_395 || continue
        df_map.map_file[i] = MagNav.ottawa_area_maps(df_map.map_name[i])
    end
    global mapS     = get_map(:Eastern_395,df_map)
    global map_used = "Eastern_395 (high-res survey)"
catch e
    @warn("Eastern_395 unavailable, using NAMAD",e)
    global mapS     = get_map(MagNav.namad)
    global map_used = "NAMAD (coarse)"
end
println("map: ",map_used)

##* simulated flight with heading variation over the map
try
    global xyz = create_XYZ0(mapS; alt=mapS.alt, dt=0.1, t=240, v=68,
                             N_waves=2, attempts=500, cor_sigma=1.0,
                             fogm_sigma=1.0, silent=true)
catch e
    @warn("trajectory fit on survey map failed, using NAMAD",e)
    global mapS     = get_map(MagNav.namad)
    global map_used = "NAMAD (coarse)"
    global xyz = create_XYZ0(mapS; alt=1000, dt=0.1, t=600, v=68, N_waves=3,
                             cor_sigma=1.0, fogm_sigma=1.0, silent=true)
end
traj = xyz.traj
ins  = xyz.ins
N    = traj.N
itp_mapS  = map_interpolate(mapS)
date      = get_years(2023,154)
mag_clean = xyz.mag_1_c

##* sensor-field geometry (optical axis = body-x / nose)
axis   = [1.0,0.0,0.0]
u_body = zeros(3,N)
theta  = zeros(N)
for t = 1:N
    v           = MagNav.igrf(date,traj.alt[t],traj.lat[t],traj.lon[t],Val(:geodetic))
    u_body[:,t] = traj.Cnb[:,:,t]' * (v ./ norm(v))
    theta[t]    = acos(clamp(dot(axis,u_body[:,t]),-1,1))
end

##* inject KNOWN sensor errors ------------------------------------------------
# OPM heading error: cosine harmonics at orders {1,2,4} (Hager et al. 2026 eq. 20;
# Wang et al. 2020, IEEE GRSL). c1 = light shift, c2 = nonlinear Zeeman, c4 higher.
c1_true = 10.0   # cos(1 ψ)  [nT]
c2_true =  6.0   # cos(2 ψ)  [nT]
c4_true =  3.0   # cos(4 ψ)  [nT]
d_head  = c1_true .* cos.(theta) .+ c2_true .* cos.(2 .* theta) .+
          c4_true .* cos.(4 .* theta)

m_true  = [15.0,-10.0,6.0]                       # hard-iron bias [nT]
d_bias  = [dot(m_true,u_body[:,t]) for t = 1:N]

dr_true = 0.02                                   # drift [nT/s]
d_drift = dr_true .* (0:N-1) .* traj.dt

# OPM equatorial dead zone (Hager et al. 2026 eq. 21): effective noise amplified
# by 1/max(|cos ψ|, ε), ε = 0.1; near ψ = 90° the OPM signal collapses.
eps_dz   = 0.1
amp      = 1.0 ./ max.(abs.(cos.(theta)), eps_dz)
dz_noise = amp .* randn(N)                        # heteroscedastic OPM noise

mag_corrupt = mag_clean .+ d_head .+ d_bias .+ d_drift .+ dz_noise
println("θ span: ",round(rad2deg(maximum(theta)-minimum(theta)),digits=0)," deg")
println("injected RMS [nT]  head=",round(sqrt(mean(d_head.^2)),digits=1),
        " bias=",round(sqrt(mean(d_bias.^2)),digits=1),
        " drift=",round(sqrt(mean(d_drift.^2)),digits=1),
        " deadzone-noise=",round(sqrt(mean(dz_noise.^2)),digits=1))

##* filter model
(P0,Qd,R) = create_model(traj.dt,traj.lat[1];fogm_sigma=1.0,fogm_tau=600.0)

function drms(fr)
    dn = dlat2dn.(ins.lat .+ fr.x[1,:] .- traj.lat, traj.lat)
    de = dlon2de.(ins.lon .+ fr.x[2,:] .- traj.lon, traj.lat)
    sqrt(mean(dn.^2 .+ de.^2))
end
ins_drms = sqrt(mean(dlat2dn.(ins.lat .- traj.lat, traj.lat).^2 .+
                     dlon2de.(ins.lon .- traj.lon, traj.lat).^2))

function recovered(fr, nh, cb, dr)
    # heading states are nh cosine coefficients (orders {1,2,4} for nh=3)
    a1 = nh >= 1 ? median(fr.x[19,:]) : NaN   # cos(1 ψ) coeff (light shift)
    a2 = nh >= 2 ? median(fr.x[20,:]) : NaN   # cos(2 ψ) coeff (nonlinear Zeeman)
    ib = 18 + nh
    m  = cb ? median(fr.x[ib+1:ib+3,:],dims=2)[:] : fill(NaN,3)
    id = 18 + nh + (cb ? 3 : 0)
    d  = dr ? median(fr.x[id+1,:]) : NaN
    return (a1,a2,m,d)
end

##* factorial ablation --------------------------------------------------------
cases = [ # name                     n_harm cal_bias drift dead_zone
    ("baseline (no sensor)",              0, false, false, false),
    ("heading (OPM, n={1,2,4})",          3, false, false, false),
    ("heading + dead-zone wt",            3, false, false, true ),
    ("heading + dz + fluxgate",           3, true,  false, true ),
    ("heading + dz + flux + drift",       3, true,  true,  true ),
]

results = DataFrame(method=String[],robust=String[],drms=Float64[],
                    a1=Float64[],a2=Float64[],mx=Float64[],my=Float64[],
                    mz=Float64[],drift=Float64[])

println("\nmap: $map_used | N=$N | INS DRMS: ",round(ins_drms,digits=1)," m")
println("truth: c1=$c1_true c2=$c2_true c4=$c4_true  m=$m_true  drift=$dr_true\n")

for robust in (:none, :huber)
    println("=== robust = $robust ===")
    for (name,nh,cb,dr,dz) in cases
        fr = fgo_sensor(ins,mag_corrupt,itp_mapS;P0=P0,Qd=Qd,R=R,core=false,
                        n_harm=nh,heading=:geometry,axis=axis,cal_bias=cb,
                        drift=dr,dead_zone=dz,robust=robust,n_iter=8)
        d  = drms(fr)
        (a1,a2,m,dd) = recovered(fr,nh,cb,dr)
        push!(results,(name,String(robust),d,a1,a2,m[1],m[2],m[3],dd))
        println(rpad(name,30)," DRMS=",rpad(round(d,digits=1),7)," m",
                nh>=1 ? "  ĉ1=$(round(a1,digits=1)) ĉ2=$(round(a2,digits=1))" : "",
                cb    ? "  m̂=$(round.(m,digits=1))" : "",
                dr    ? "  d̂=$(round(dd,digits=3))" : "")
    end
    println()
end

println("=== factorial results ===")
show(results;allrows=true,allcols=true)
println()
out_csv = joinpath(@__DIR__,"fgo_sensor_ablation_results.csv")
CSV.write(out_csv,results)
println("\nresults written to $out_csv")
