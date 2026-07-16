##* Does the non-TL remainder align with platform currents? (mechanism check)
#
# The paper claims the cold-start FGO absorbs the non-Tolles-Lawson remainder with
# (i) the time-varying beta tracking its low-frequency projection, (ii) the FOGM
# disturbance state S carrying the correlated leftover, and (iii) the Huber kernel
# attenuating switching transients. This script checks that physical story against
# the SGL 2020 current sensors (cur_*), which record exactly the non-attitude
# sources the TL basis cannot span:
#
#   1. How much of the RAW cabin interference (mag_uc - mag_1_c) is linearly
#      explained by the current channels?  (how current-driven is the problem)
#   2. How much of the POST-FIT windowed-FGO residual still is?  (how much the
#      graph's beta+S already absorbed)
#   3. Do the Huber-downweighted epochs coincide with current activity
#      (large |d cur/dt|)?  (is the kernel attenuating current switching)
#
# Line 1003.02 (Eastern map), cabin Mag 4 and Mag 5 — same setup as fgo_breadth.jl.
# Outputs research/fgo_current_corr.csv. Usage:
#   julia --project=. research/fgo_current_corr.jl

using MagNav
using CSV, DataFrames
using LinearAlgebra, Statistics
using Random: seed!
seed!(33)

MEAS_VAR = 12.0^2
FOGM_SIG = 3.0
FOGM_TAU = 180.0
TERMS    = [:permanent,:induced,:eddy,:bias]
HUBER_C  = 1.345          # MagNav.jl default (src/fgo.jl)
FLIGHT   = :Flt1003
LINE     = 1003.02

CURS = (:cur_com_1,:cur_ac_hi,:cur_ac_lo,:cur_tank,:cur_flap,:cur_strb,
        :cur_srvo_o,:cur_srvo_m,:cur_srvo_i,:cur_heat,:cur_acpwr,:cur_outpwr,
        :cur_bat_1,:cur_bat_2)

##* data ------------------------------------------------------------------------
df_dir    = joinpath(@__DIR__,"..","examples","dataframes")
df_flight = DataFrame(CSV.File(joinpath(df_dir,"df_flight.csv")))
df_flight[!,:flight]   = Symbol.(df_flight[!,:flight])
df_flight[!,:xyz_type] = Symbol.(df_flight[!,:xyz_type])
df_flight[!,:xyz_file] = String.(df_flight[!,:xyz_file])
for (i,fl) in enumerate(df_flight.flight)
    fl == FLIGHT || continue
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

xyz   = get_XYZ(FLIGHT,df_flight;silent=true)
ind   = get_ind(xyz,LINE,df_nav)
mname = df_nav[(df_nav.flight.==FLIGHT).&(df_nav.line.==LINE),:map_name][1]
mapS  = get_map(mname,df_map)
traj  = get_traj(xyz,ind)
ins   = get_ins(xyz,ind;N_zero_ll=1)
(_,itp) = get_map_val(mapS,traj;return_itp=true)
flux  = xyz.flux_d(ind)
nTL   = size(create_TL_A(flux;terms=TERMS),2)
(P0,Qd,R) = create_model(traj.dt,traj.lat[1];init_pos_sigma=0.1,init_alt_sigma=1.0,
    init_vel_sigma=1.0,meas_var=MEAS_VAR,fogm_sigma=FOGM_SIG,fogm_tau=FOGM_TAU,
    vec_states=false,TL_sigma=fill(1.0,nTL),P0_TL=Matrix(Diagonal(fill(1.0,nTL))))

zstd(v) = (s = std(v); s > 0 ? (v .- mean(v))./s : zero(v))
# smoothed |d cur/dt| switching-activity trace (2 s boxcar)
function activity(v,dt)
    a = [0.0; abs.(diff(v))]
    w = max(1,round(Int,2.0/dt))
    [mean(@view a[max(1,i-w+1):i]) for i in eachindex(a)]
end
r2_on(y,X) = (Xa = [ones(length(y)) X]; e = y - Xa*(Xa\y);
              1 - sum(abs2,e)/sum(abs2,y .- mean(y)))

curmat = reduce(hcat,[Float64.(getfield(xyz,c)[ind]) for c in CURS])
actmat = reduce(hcat,[activity(curmat[:,j],traj.dt) for j in axes(curmat,2)])
act_any = vec(maximum(reduce(hcat,[zstd(actmat[:,j]) for j in axes(actmat,2)]),dims=2)) .> 2

rows = DataFrame(mag=String[],channel=String[],corr_intf=Float64[],
                 corr_resid=Float64[],corr_resid_act=Float64[])
summ = DataFrame(mag=String[],R2_intf=Float64[],R2_resid=Float64[],
                 p_down=Float64[],p_down_active=Float64[],p_active=Float64[],
                 drms=Float64[])

for magsym in (:mag_4_uc,:mag_5_uc)
    mag = getfield(xyz,magsym)[ind]
    tag = replace(String(magsym),"mag_"=>"Mag ","_uc"=>"")
    intf = mag .- xyz.mag_1_c[ind]           # raw cabin interference proxy

    fr = fgo_online(ins,mag,flux,itp,zeros(nTL),P0,Qd,R;terms=TERMS,
                    core=true,win=300.0,overlap=90.0,robust=:huber)
    resid = vec(fr.r)
    fo = MagNav.eval_filt(traj,ins,fr)
    m  = (traj.tt .- traj.tt[1]) .>= 600.0
    dn = dlat2dn.(fo.lat[m].-traj.lat[m],traj.lat[m])
    de = dlon2de.(fo.lon[m].-traj.lon[m],traj.lat[m])
    drms = sqrt(mean(dn.^2 .+ de.^2))

    w = min.(1.0, HUBER_C ./ max.(abs.(resid)./sqrt(MEAS_VAR), eps()))
    down = w .< 1.0
    for (j,c) in enumerate(CURS)
        push!(rows,(tag,String(c),
            round(cor(zstd(intf),  zstd(curmat[:,j])),digits=3),
            round(cor(zstd(resid), zstd(curmat[:,j])),digits=3),
            round(cor(zstd(abs.(resid)),zstd(actmat[:,j])),digits=3)))
    end
    push!(summ,(tag,round(r2_on(intf,curmat),digits=3),
                round(r2_on(resid,curmat),digits=3),
                round(mean(down),digits=3),
                round(sum(down .& act_any)/max(sum(act_any),1),digits=3),
                round(mean(act_any),digits=3),round(drms,digits=1)))
    println("$tag  DRMS=$(round(drms,digits=1)) m  ",
            "R2(currents→raw intf)=$(summ.R2_intf[end])  ",
            "R2(currents→FGO resid)=$(summ.R2_resid[end])  ",
            "P(down)=$(summ.p_down[end])  P(down|active)=$(summ.p_down_active[end])")
end

println("\n=== per-channel correlations (top |corr| with raw interference) ===")
show(sort(rows,order(:corr_intf,by=abs,rev=true));allrows=true,allcols=true); println()
CSV.write(joinpath(@__DIR__,"fgo_current_corr.csv"),rows)
CSV.write(joinpath(@__DIR__,"fgo_current_corr_summary.csv"),summ)
println("wrote fgo_current_corr.csv, fgo_current_corr_summary.csv")
