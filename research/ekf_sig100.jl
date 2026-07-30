##* Online-TL EKF with the proposed estimator's compensation prior.
#
# The breadth-table EKF runs used the reference implementation's P0_TL = I
# (sigma_beta = 1). The proposed estimator declares sigma_beta = 100. This
# runs the online-TL EKF at sigma_beta = 100 (and 700) on all 8 counted cases,
# everything else at the fgo_breadth.jl configuration, so the comparison
# cannot be attributed to the prior: does the wider prior rescue or degrade
# the recursive filter?
#
# Usage: julia --project=. research/ekf_sig100.jl
# Outputs research/ekf_sig100_results.csv

using MagNav
using CSV, DataFrames
using LinearAlgebra, Statistics
using Random: seed!
seed!(33)

FOGM_SIG = 3.0
FOGM_TAU = 180.0
TERMS    = [:permanent,:induced,:eddy,:bias]
MEAS_VAR = 12.0^2
SIGMAS   = [100.0, 700.0]
LINES = [(:Flt1007,1007.06), (:Flt1007,1007.02),
         (:Flt1003,1003.02), (:Flt1003,1003.08)]
flights = unique(first.(LINES))

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
for (i,mn) in enumerate(df_map.map_name)
    df_map.map_file[i] = MagNav.ottawa_area_maps(mn)
end
df_nav = DataFrame(CSV.File(joinpath(df_dir,"df_nav.csv")))
df_nav[!,:flight]   = Symbol.(df_nav[!,:flight])
df_nav[!,:map_name] = Symbol.(df_nav[!,:map_name])

function drms(traj, lat, lon; warm=600.0, div_thresh=1e4)
    m = (traj.tt .- traj.tt[1]) .>= warm
    d = sqrt(mean(dlat2dn.(lat[m] .- traj.lat[m], traj.lat[m]).^2 .+
                  dlon2de.(lon[m] .- traj.lon[m], traj.lat[m]).^2))
    (isfinite(d) && d < div_thresh) ? d : Inf
end

results = DataFrame(flight=Symbol[],line=Float64[],mag=String[],
                    sigma_beta=Float64[],EKF_online=Float64[])

for (fl,line) in LINES
    xyz  = get_XYZ(fl,df_flight;silent=true)
    ind  = get_ind(xyz,line,df_nav)
    mname= df_nav[(df_nav.flight.==fl).&(df_nav.line.==line),:map_name][1]
    mapS = get_map(mname,df_map)
    traj = get_traj(xyz,ind)
    ins  = get_ins(xyz,ind;N_zero_ll=1)
    (_,itp) = get_map_val(mapS,traj;return_itp=true)
    flux = xyz.flux_d(ind)
    nTL  = size(create_TL_A(flux;terms=TERMS),2)

    for magsym in (:mag_4_uc,:mag_5_uc)
        mag = getfield(xyz,magsym)[ind]
        tag = replace(String(magsym),"mag_"=>"Mag ","_uc"=>"")
        for sb in SIGMAS
            (P0,Qd,R) = create_model(traj.dt,traj.lat[1];init_pos_sigma=0.1,
                init_alt_sigma=1.0,init_vel_sigma=1.0,meas_var=MEAS_VAR,
                fogm_sigma=FOGM_SIG,fogm_tau=FOGM_TAU,vec_states=false,
                TL_sigma=fill(1.0,nTL),P0_TL=Matrix(Diagonal(fill(sb^2,nTL))))
            d_ekf = try
                fo = run_filt(traj,ins,mag,itp,:ekf_online;P0=P0,Qd=Qd,R=R,
                              flux=flux,x0_TL=zeros(nTL),terms=TERMS,
                              core=true,run_crlb=false)
                drms(traj,fo.lat,fo.lon)
            catch e; @warn("ekf_online",e); Inf end
            push!(results,(fl,line,tag,sb,round(d_ekf,digits=1)))
            println("$fl $line $tag  sigma_beta=$sb  EKF-online=",
                    "$(round(d_ekf,digits=1)) m"); flush(stdout)
            CSV.write(joinpath(@__DIR__,"ekf_sig100_results.csv"),results)
        end
    end
end
println("done")
