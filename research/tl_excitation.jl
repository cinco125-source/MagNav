##* TL-excitation timeline: does the cold start rely on early maneuvers?
#
# Reviewer question this answers: "your window works because calibration-rich
# maneuvers happen early -- what does the attitude excitation actually look like
# over each line?" For every counted breadth line we slide the same 5-min window
# the estimator uses and record, per window:
#   - sigma_min / cond of the column-whitened TL basis (local identifiability of
#     the coefficients from that window alone)
#   - yaw spread and roll std (raw attitude excitation)
# and summarize: first-window excitation vs the line median, and the time of the
# first window whose sigma_min reaches half the line median ("first excited
# window"). If the first window is already near the median, excitation is
# distributed rather than event-like, and no special early maneuver is doing the
# work.
#
# Outputs research/tl_excitation.csv (per window) and
# research/tl_excitation_summary.csv. Usage:
#   julia --project=. research/tl_excitation.jl

using MagNav
using CSV, DataFrames
using LinearAlgebra, Statistics

TERMS = [:permanent,:induced,:eddy,:bias]
WIN   = 300.0    # [s], same as the estimator's window
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
df_nav = DataFrame(CSV.File(joinpath(df_dir,"df_nav.csv")))
df_nav[!,:flight]   = Symbol.(df_nav[!,:flight])
df_nav[!,:map_name] = Symbol.(df_nav[!,:map_name])

xyz_cache = Dict{Symbol,Any}()
getxyz(fl) = get!(()->get_XYZ(fl,df_flight;silent=true), xyz_cache, fl)

unwrap(a) = (out = copy(a); for i in 2:length(out)
    d = out[i]-out[i-1]; d >  pi && (out[i:end] .-= 2pi); d < -pi && (out[i:end] .+= 2pi)
end; out)

rows = DataFrame(flight=Symbol[],line=Float64[],t_min=Float64[],
                 sigma_min=Float64[],cond=Float64[],
                 yaw_spread_deg=Float64[],roll_std_deg=Float64[])
summ = DataFrame(flight=Symbol[],line=Float64[],n_win=Int[],
                 smin_first=Float64[],smin_median=Float64[],
                 first_excited_min=Float64[],
                 yaw_first_deg=Float64[],yaw_median_deg=Float64[])

for (fl,line) in LINES
    xyz  = getxyz(fl)
    ind  = get_ind(xyz,line,df_nav)
    traj = get_traj(xyz,ind)
    flux = xyz.flux_d(ind)
    A    = create_TL_A(flux;terms=TERMS)
    # whiten columns over the line so sigma_min is scale-free across term groups
    Az   = (A .- mean(A,dims=1)) ./ max.(std(A,dims=1), 1e-12)
    yaw  = unwrap(Float64.(xyz.ins_yaw[ind]))
    roll = Float64.(xyz.ins_roll[ind])
    Lw   = round(Int, WIN/traj.dt)
    N    = traj.N
    smin = Float64[]; tmins = Float64[]
    for i0 in 1:Lw:(N-Lw+1)
        w  = i0:(i0+Lw-1)
        s  = svdvals(Az[w,:])
        ys = rad2deg(maximum(yaw[w]) - minimum(yaw[w]))
        rs = rad2deg(std(roll[w]))
        tm = (traj.tt[i0]-traj.tt[1])/60
        push!(rows,(fl,line,round(tm,digits=1),round(s[end],digits=2),
                    round(s[1]/max(s[end],1e-12),digits=1),
                    round(ys,digits=1),round(rs,digits=2)))
        push!(smin,s[end]); push!(tmins,tm)
    end
    med = median(smin)
    fe  = findfirst(>=(0.5*med), smin)
    yawcol = rows[(rows.flight.==fl).&(rows.line.==line),:yaw_spread_deg]
    push!(summ,(fl,line,length(smin),round(smin[1],digits=2),round(med,digits=2),
                fe === nothing ? NaN : round(tmins[fe],digits=1),
                yawcol[1],round(median(yawcol),digits=1)))
    println("$fl $line: $(length(smin)) windows | sigma_min first=$(round(smin[1],digits=2)) ",
            "median=$(round(med,digits=2)) | first excited window at ",
            "$(fe === nothing ? "never" : string(round(tmins[fe],digits=1)))"," min | ",
            "yaw spread first=$(yawcol[1]) deg median=$(round(median(yawcol),digits=1)) deg")
end

println("\n=== TL excitation per 5-min window ===")
show(rows;allrows=true,allcols=true); println()
CSV.write(joinpath(@__DIR__,"tl_excitation.csv"),rows)
CSV.write(joinpath(@__DIR__,"tl_excitation_summary.csv"),summ)
println("wrote tl_excitation.csv, tl_excitation_summary.csv")
