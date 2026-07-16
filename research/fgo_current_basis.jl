##* Current-augmented compensation basis: instrumentation instead of learning.
#
# fgo_current_corr.jl showed the recorded platform currents linearly explain
# 28-39% of the raw cabin interference. A current loop is fixed aircraft geometry,
# so its field at the sensor is m_i * I_i(t) (body-frame constant vector times the
# measured current), and its scalar projection is (b_hat . m_i) I_i(t) -- exactly a
# Tolles-Lawson permanent term whose "magnetization" is switched by a MEASURED
# telemetry channel. This script augments the online TL basis with the current
# channels (fgo_online's A_extra hook) and asks whether the cold-start navigation
# improves. The currents are exogenous (position-independent onboard telemetry),
# so the joint-observability corollary sanctions the enlargement.
#
# Configs (ablation):
#   A  FGO-win (TL only)                    -- breadth baseline, rerun here
#   B  + currents, random-walk gamma        -- gamma treated like TL beta
#   C  + currents, static gamma             -- physical: coupling geometry is fixed
#   D  + currents & top-3 (u,v,w)-modulated -- attitude-projected triplets, static
#
# Channel handling: causal validity filter (first-10-min std), causal z-scoring
# (first-10-min mean/std) so the cold-start story stays honest; no
# correlation-based selection (avoids selection bias).
#
# Cross-line replication: for config C the estimated physical couplings
# gamma_i [nT/A] are saved per line; agreement across lines of the same aircraft
# is the falsifiable check that the coupling is geometry, not overfitting.
#
# Outputs research/fgo_current_basis_results.csv and research/fgo_current_gamma.csv.
# Usage: julia --project=. research/fgo_current_basis.jl

using MagNav
using CSV, DataFrames
using LinearAlgebra, Statistics
using Random: seed!
seed!(33)

MEAS_VAR = 12.0^2
FOGM_SIG = 3.0
FOGM_TAU = 180.0
TERMS    = [:permanent,:induced,:eddy,:bias]
WARM     = 600.0
LINES = [(:Flt1003,1003.02), (:Flt1003,1003.08),
         (:Flt1007,1007.02), (:Flt1007,1007.06)]
flights = unique(first.(LINES))

CURS = (:cur_com_1,:cur_ac_hi,:cur_ac_lo,:cur_tank,:cur_flap,:cur_strb,
        :cur_srvo_o,:cur_srvo_m,:cur_srvo_i,:cur_heat,:cur_acpwr,:cur_outpwr,
        :cur_bat_1,:cur_bat_2)

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

xyz_cache = Dict{Symbol,Any}()
getxyz(fl) = get!(()->get_XYZ(fl,df_flight;silent=true), xyz_cache, fl)

function drms_ll(traj, lat, lon; warm=WARM)
    m  = (traj.tt .- traj.tt[1]) .>= warm
    dn = dlat2dn.(lat[m] .- traj.lat[m], traj.lat[m])
    de = dlon2de.(lon[m] .- traj.lon[m], traj.lat[m])
    sqrt(mean(dn.^2 .+ de.^2))
end

results = DataFrame(flight=Symbol[],line=Float64[],mag=String[],config=String[],
                    n_extra=Int[],DRMS=Float64[],t=Float64[])
gammas  = DataFrame(flight=Symbol[],line=Float64[],mag=String[],channel=String[],
                    gamma_nT_per_A=Float64[])

for (fl,line) in LINES
    xyz   = getxyz(fl)
    ind   = get_ind(xyz,line,df_nav)
    mname = df_nav[(df_nav.flight.==fl).&(df_nav.line.==line),:map_name][1]
    mapS  = get_map(mname,df_map)
    traj  = get_traj(xyz,ind)
    ins   = get_ins(xyz,ind;N_zero_ll=1)
    (_,itp) = get_map_val(mapS,traj;return_itp=true)
    flux  = xyz.flux_d(ind)
    nTL   = size(create_TL_A(flux;terms=TERMS),2)
    N     = traj.N
    n10   = min(N, round(Int, WARM/traj.dt))   # causal (first 10 min) stats

    # causal channel validity + z-scoring
    chans = Symbol[]; cols = Vector{Float64}[]; scales = Float64[]
    for c in CURS
        v = Float64.(getfield(xyz,c)[ind])
        any(isnan,v) && continue
        mu = mean(@view v[1:n10]); sd = std(@view v[1:n10])
        (isfinite(sd) && sd > 1e-4) || continue     # constant/dead channel
        push!(chans,c); push!(cols,(v .- mu)./sd); push!(scales,sd)
    end
    n_ch = length(chans)
    if n_ch == 0
        @warn("$fl $line: no valid current channels; skipping line")
        continue
    end
    U = reduce(hcat,cols)                           # N x n_ch, unit-variance

    # field direction cosines in body frame for the (u,v,w)-modulated triplets
    Bt = sqrt.(flux.x.^2 .+ flux.y.^2 .+ flux.z.^2)
    u = flux.x./Bt; v = flux.y./Bt; w = flux.z./Bt
    top3 = sortperm([std(@view U[1:n10,j]) for j in 1:n_ch], rev=true)[1:min(3,n_ch)]
    Utrip = reduce(hcat,[U[:,j].*d for j in top3 for d in (u,v,w)])

    println("\n$fl $line ($mname): $n_ch valid channels ",
            "($(join(String.(chans),","))); triplets on $(join(String.(chans[top3]),","))")

    # (config, A_extra, sigma_extra, P0_extra)
    CFG = [("A TL only",        nothing,          Float64[],            Float64[]),
           ("B +cur RW",        U,                fill(1.0,n_ch),       fill(1.0,n_ch)),
           ("C +cur static",    U,                fill(1e-6,n_ch),      fill(900.0,n_ch)),
           ("D +cur+triplets",  hcat(U,Utrip),    fill(1e-6,n_ch+3*length(top3)),
                                                  fill(900.0,n_ch+3*length(top3)))]

    for magsym in (:mag_4_uc,:mag_5_uc)
        mag = getfield(xyz,magsym)[ind]
        tag = replace(String(magsym),"mag_"=>"Mag ","_uc"=>"")
        for (name,Ax,sig_x,P0_x) in CFG
            nx_x  = length(sig_x)
            sigs  = vcat(fill(1.0,nTL), sig_x)
            P0d   = vcat(fill(1.0,nTL), P0_x)
            (P0,Qd,R) = create_model(traj.dt,traj.lat[1];init_pos_sigma=0.1,
                init_alt_sigma=1.0,init_vel_sigma=1.0,meas_var=MEAS_VAR,
                fogm_sigma=FOGM_SIG,fogm_tau=FOGM_TAU,vec_states=false,
                TL_sigma=sigs,P0_TL=Matrix(Diagonal(P0d)))
            d = NaN; tel = NaN
            try
                tel = @elapsed begin
                    fr = fgo_online(ins,mag,flux,itp,zeros(nTL+nx_x),P0,Qd,R;
                                    terms=TERMS,core=true,win=300.0,overlap=90.0,
                                    robust=:huber,A_extra=Ax)
                    fo = MagNav.eval_filt(traj,ins,fr)
                    d  = drms_ll(traj,fo.lat,fo.lon)
                    if name == "C +cur static"   # physical couplings, final estimate
                        nlast = max(1, N - round(Int,300/traj.dt))
                        for (j,c) in enumerate(chans)
                            g = median(@view fr.x[17+nTL+j, nlast:N]) / scales[j]
                            push!(gammas,(fl,line,tag,String(c),round(g,digits=3)))
                        end
                    end
                end
            catch e
                @warn("$fl $line $tag $name failed",e)
            end
            push!(results,(fl,line,tag,name,nx_x,round(d,digits=1),round(tel,digits=2)))
            println("  $tag  $(rpad(name,16)) (+$nx_x states)  DRMS=$(round(d,digits=1)) m",
                    "  ($(round(tel,digits=1)) s)")
        end
    end
end

println("\n=== current-augmented basis (ablation) ===")
show(results;allrows=true,allcols=true); println()
println("\n=== estimated physical couplings, config C [nT/A] ===")
show(gammas;allrows=true,allcols=true); println()
CSV.write(joinpath(@__DIR__,"fgo_current_basis_results.csv"),results)
CSV.write(joinpath(@__DIR__,"fgo_current_gamma.csv"),gammas)
println("wrote fgo_current_basis_results.csv, fgo_current_gamma.csv")
