##* MPF+TL cold-start: does raising R (or the particle count) rescue it?
#
# The breadth MPF_TL column is Inf on every full line: the RBPF diverges from a cold
# start. The hardened weight update (log-domain + geometric R warm-up in
# mpf_online.jl) removed the numerical underflow, but the DRMS stayed useless. The
# open question is WHICH limiter bites:
#   (a) too much map trust early  -> raising the nominal measurement variance R
#       should help, or
#   (b) particle depletion in the position (nonlinear) block -> raising R does
#       nothing and only more particles could help (and even that may not).
#
# We sweep a nominal-R multiplier at the breadth particle count, then a particle
# probe at a middling R, on the WELL-CONDITIONED Mag 5 lines (the best case for the
# MPF; if higher R cannot rescue Mag 5 it cannot rescue Mag 4). Mag 4 on the primary
# line is included as the hard reference.
#
# Usage: julia --project=. research/mpf_R_sweep.jl

using MagNav
using CSV, DataFrames
using LinearAlgebra, Statistics
using Random: seed!
seed!(33)

include(joinpath(@__DIR__,"mpf_online.jl"))

MEAS_VAR = 12.0^2
FOGM_SIG = 3.0
FOGM_TAU = 180.0
TERMS    = [:permanent,:induced,:eddy,:bias]

LINES  = [(:Flt1003,1003.02), (:Flt1003,1003.08), (:Flt1007,1007.06)]
R_MULT = [1.0, 4.0, 16.0, 64.0, 256.0]      # nominal measurement-variance multiplier
NP_PROBE = [(300, 16.0), (2000, 16.0)]       # (num_part, R_mult) depletion probe
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
for (i,map_name) in enumerate(df_map.map_name)
    df_map.map_file[i] = MagNav.ottawa_area_maps(map_name)
end
df_nav = DataFrame(CSV.File(joinpath(df_dir,"df_nav.csv")))
df_nav[!,:flight]   = Symbol.(df_nav[!,:flight])
df_nav[!,:map_name] = Symbol.(df_nav[!,:map_name])

results = DataFrame(flight=Symbol[],line=Float64[],mag=String[],
                    probe=String[],num_part=Int[],R_mult=Float64[],DRMS=Float64[])

function run_line(fl,line,magsyms)
    xyz  = get_XYZ(fl,df_flight;silent=true)
    ind  = get_ind(xyz,line,df_nav)
    mname= df_nav[(df_nav.flight.==fl).&(df_nav.line.==line),:map_name][1]
    mapS = get_map(mname,df_map)
    traj = get_traj(xyz,ind)
    ins  = get_ins(xyz,ind;N_zero_ll=1)
    (_,itp) = get_map_val(mapS,traj;return_itp=true)
    flux = xyz.flux_d(ind)
    nTL  = size(create_TL_A(flux;terms=TERMS),2)
    (P0,Qd,R) = create_model(traj.dt,traj.lat[1];init_pos_sigma=0.1,init_alt_sigma=1.0,
        init_vel_sigma=1.0,meas_var=MEAS_VAR,fogm_sigma=FOGM_SIG,fogm_tau=FOGM_TAU,
        vec_states=false,TL_sigma=fill(1.0,nTL),P0_TL=Matrix(Diagonal(fill(1.0,nTL))))
    println("\n$fl $line ($mname, ~$(round(traj.N*traj.dt/60,digits=0)) min)")
    for magsym in magsyms
        mag = getfield(xyz,magsym)[ind]
        tag = replace(String(magsym),"mag_"=>"Mag ","_uc"=>"")
        for rm in R_MULT
            d = mpf_online_drms(traj,ins,mag,flux,itp,zeros(nTL),P0,Qd,R.*rm;
                                terms=TERMS,num_part=300)
            push!(results,(fl,line,tag,"R_sweep",300,rm,round(d,digits=1)))
            println("  $tag  R×$(rm)  NP=300   DRMS=$(round(d,digits=1)) m")
        end
        if magsym == :mag_5_uc
            for (npart,rm) in NP_PROBE
                d = mpf_online_drms(traj,ins,mag,flux,itp,zeros(nTL),P0,Qd,R.*rm;
                                    terms=TERMS,num_part=npart)
                push!(results,(fl,line,tag,"NP_probe",npart,rm,round(d,digits=1)))
                println("  $tag  R×$(rm)  NP=$(npart)  DRMS=$(round(d,digits=1)) m")
            end
        end
    end
end

for (fl,line) in LINES
    magsyms = (fl,line) == (:Flt1007,1007.06) ? (:mag_5_uc,:mag_4_uc) : (:mag_5_uc,)
    try
        run_line(fl,line,magsyms)
    catch e
        @warn("run_line failed $fl $line", e)
    end
end

CSV.write(joinpath(@__DIR__,"mpf_R_sweep_results.csv"),results)
println("\n=== MPF+TL R / particle sweep ===")
show(results, allrows=true, allcols=true); println()
println("wrote mpf_R_sweep_results.csv")
