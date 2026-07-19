##* MPF clean-signal control: is the RBPF sound, or under-powered?
#
# The MPF fails on the uncompensated cabin magnetometers. Is that because the PF
# machinery is weak (too few particles / poor proposal), or because it is sensitive
# to the residual measurement-model error (leftover interference)? This control runs
# the SAME MPF on the COMPENSATED stinger mag_1_c, where the interference is already
# removed, so the measurement matches the map. EKF and FGO on the same clean signal
# are the reference. If MPF is bounded on mag_1_c but diverges on mag_5_uc, the
# failure is sensitivity to residual interference, not the sampler itself.
#
# Usage: julia --project=. research/mpf_clean.jl

using MagNav
using CSV, DataFrames
using LinearAlgebra, Statistics
using Random: seed!
seed!(33)

include(joinpath(@__DIR__,"mpf_online.jl"))
const MPF_NP = 300

MEAS_VAR = 12.0^2
FOGM_SIG = 3.0
FOGM_TAU = 180.0
TERMS    = [:permanent,:induced,:eddy,:bias]

LINES = [(:Flt1003,1003.02), (:Flt1003,1003.08),
         (:Flt1007,1007.02), (:Flt1007,1007.06)]
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

function drms(traj, lat, lon; warm=600.0)
    m = (traj.tt .- traj.tt[1]) .>= warm
    sqrt(mean(dlat2dn.(lat[m] .- traj.lat[m], traj.lat[m]).^2 .+
              dlon2de.(lon[m] .- traj.lon[m], traj.lat[m]).^2))
end

results = DataFrame(flight=Symbol[],line=Float64[],signal=String[],method=String[],DRMS=Float64[])

for (fl,line) in LINES
    xyz  = get_XYZ(fl,df_flight;silent=true)
    ind  = get_ind(xyz,line,df_nav)
    mname= df_nav[(df_nav.flight.==fl).&(df_nav.line.==line),:map_name][1]
    mapS = get_map(mname,df_map)
    traj = get_traj(xyz,ind)
    ins  = get_ins(xyz,ind;N_zero_ll=1)
    (_,itp) = get_map_val(mapS,traj;return_itp=true)   # cubic (EKF/FGO reference)
    itp_lin = map_interpolate(mapS, :linear)           # linear: graceful out-of-grid for the PF
    flux = xyz.flux_d(ind)
    nTL  = size(create_TL_A(flux;terms=TERMS),2)
    (P0,Qd,R)    = create_model(traj.dt,traj.lat[1];init_pos_sigma=3.0,init_alt_sigma=1.0,
        init_vel_sigma=1.0,meas_var=MEAS_VAR,fogm_sigma=FOGM_SIG,fogm_tau=FOGM_TAU,
        vec_states=false,TL_sigma=fill(1.0,nTL),P0_TL=Matrix(Diagonal(fill(1.0,nTL))))
    (P0n,Qdn,Rn) = create_model(traj.dt,traj.lat[1];init_pos_sigma=3.0,init_alt_sigma=1.0,
        init_vel_sigma=1.0,meas_var=MEAS_VAR,fogm_sigma=FOGM_SIG,fogm_tau=FOGM_TAU)

    clean = xyz.mag_1_c[ind]          # compensated stinger — interference removed
    dirty = xyz.mag_5_uc[ind]         # uncompensated cabin mag — full interference

    ekf_d(mag) = try drms(traj, (fo=run_filt(traj,ins,mag,itp,:ekf;P0=P0n,Qd=Qdn,R=Rn,
                    core=true,run_crlb=false)).lat, fo.lon) catch e; @warn("EKF",e); Inf end
    fgo_d(mag) = try drms(traj, (fo=run_filt(traj,ins,mag,itp,:fgo;P0=P0n,Qd=Qdn,R=Rn,
                    core=true,run_crlb=false)).lat, fo.lon) catch e; @warn("FGO",e); Inf end
    mpf_d(mag) = mpf_online_drms(traj,ins,mag,flux,itp_lin,zeros(nTL),P0,Qd,R;
                    terms=TERMS,num_part=MPF_NP)

    # native (vanilla) MagNav.mpf on the LINEAR itp (graceful out-of-grid)
    native_d(mag, np) = try
        fr = MagNav.mpf(ins, mag, itp_lin; P0=P0n, Qd=Qdn, R=Rn, num_part=np, core=true)
        fr.c ? drms(traj, (fo=MagNav.eval_filt(traj,ins,fr)).lat, fo.lon) : Inf
    catch e; @warn("native mpf", e); Inf end

    println("\n$fl $line ($mname)")
    for (sig, mag) in (("Mag1 (comp)", clean),)
        for (m, d) in (("EKF", ekf_d(mag)), ("FGO", fgo_d(mag)), ("MPF-ours", mpf_d(mag)),
                       ("MPF-native-300", native_d(mag,300)),
                       ("MPF-native-1000", native_d(mag,1000)))
            push!(results,(fl,line,sig,m,round(d,digits=1)))
            println("  $sig  $m  DRMS=$(round(d,digits=1)) m")
        end
    end
    push!(results,(fl,line,"Mag5 (uncomp)","MPF-ours",round(mpf_d(dirty),digits=1)))
    println("  Mag5 (uncomp)  MPF-ours  DRMS done")

    # Gnadt canonical default model (create_P0 / create_Qd defaults) instead of our
    # create_model, full line, linear itp. Tries R = Gnadt-default 1.0 and our 144.
    for (rlab, rr) in (("R1", 1.0), ("R144", Rn))
        d = try
            fr = MagNav.mpf(ins, clean, itp_lin; P0=MagNav.create_P0(traj.lat[1]),
                            Qd=MagNav.create_Qd(traj.dt), R=rr, num_part=1000, core=true)
            fr.c ? drms(traj, (fo=MagNav.eval_filt(traj,ins,fr)).lat, fo.lon) : Inf
        catch e; @warn("gnadt-default mpf",e); Inf end
        push!(results,(fl,line,"Mag1 gnadt-P0Qd $rlab","MPF-native-1000",round(d,digits=1)))
        println("  Mag1 gnadt-P0Qd $rlab  MPF-native-1000  DRMS=$(round(d,digits=1)) m")
    end

    # SHORT-SEGMENT control (linear itp): does native mpf survive the first 5/10/20 min?
    # Works short + dies long => long-line particle depletion, not a config bug.
    tind = findall(ind)                       # ind is a BitVector mask over the flight
    for mins in (5.0, 10.0, 20.0)
        d = try
            N     = min(length(tind), round(Int, mins*60/traj.dt))
            ind_s = falses(length(ind)); ind_s[tind[1:N]] .= true
            traj_s = get_traj(xyz, ind_s)
            ins_s  = get_ins(xyz, ind_s; N_zero_ll=1)
            fr = MagNav.mpf(ins_s, xyz.mag_1_c[ind_s], itp_lin; P0=P0n, Qd=Qdn, R=Rn,
                            num_part=1000, core=true)
            fr.c ? drms(traj_s, (fo=MagNav.eval_filt(traj_s,ins_s,fr)).lat, fo.lon; warm=60.0) : Inf
        catch e; @warn("native mpf short",e); Inf end
        push!(results,(fl,line,"Mag1 first-$(round(Int,mins))min","MPF-native-1000",round(d,digits=1)))
        println("  Mag1 first-$(round(Int,mins))min  MPF-native-1000  DRMS=$(round(d,digits=1)) m")
    end
end

CSV.write(joinpath(@__DIR__,"mpf_clean_results.csv"), results)
println("\n=== MPF clean-signal control ===")
show(results, allrows=true, allcols=true); println()
println("wrote mpf_clean_results.csv")
