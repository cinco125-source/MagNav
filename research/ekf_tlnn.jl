##* Reusable cold-start online EKF+TL+NN baseline (recipe-level, NOT the paper's filter)
#
# ⚠ THIS IS NOT A REIMPLEMENTATION OF HAGER ET AL. 2026. It was labelled as one;
# the audit against the preprint says otherwise, in eight places, two of them
# architectural: the NN here has NO Tolles-Lawson states beside it (it sits in
# MagNav.jl's TL slot and REPLACES the linear model, where the paper's design is
# additive TL + NN-residual), and its input is 3 permanent direction cosines
# where theirs is 4 (the vector components AND the uncompensated scalar). Also
# hidden 8 vs 5, Q_NN 3e-3 vs 1e-20, P0_NN sigma 0.3 vs 1, data-driven output
# scaling vs a fixed 400 nT, R 12^2 vs their value, and no chi-square gate.
#
# The faithful build is research/hager_impl.jl. Keep this one as what it honestly
# is -- a strong, tuned, MagNav.jl-native online NN baseline -- and do not cite it
# as the paper's method.
#
# Factors the validated line-1007.06 reproduction of research/paper_impl.jl into a
# single function so the *strong* causal baseline can be run across every breadth
# line (research/fgo_breadth.jl), not just the one line the NN paper reports. The
# recipe is unchanged: attitude-only TL features (map excluded), data-driven DC
# initialization of the NN output bias, and the locked hyperparameters below.
#
# Returns horizontal DRMS [m] after a `warm`-second warm-up, or `Inf` if the
# filter diverges / errors (so the caller can mark it "div.").

using LinearAlgebra, Statistics

# locked hyperparameters (identical to research/paper_impl.jl)
const TLNN_HIDDEN   = [8]
const TLNN_TERMS    = [:permanent]
const TLNN_P0_SIGMA = 0.3
const TLNN_WEIGHT_Q = 3e-3
const TLNN_MEAS_VAR = 12.0^2
const TLNN_FOGM_SIG = 3.0
const TLNN_FOGM_TAU = 180.0

"""
    ekf_tlnn_drms(traj, ins, mag_uc, flux, itp_mapS; warm=600.0, div_thresh=1e4)

Cold-start online EKF+TL+NN horizontal DRMS [m] on one line/magnetometer.
`Inf` on divergence (DRMS > `div_thresh`) or on error.
"""
function ekf_tlnn_drms(traj, ins, mag_uc, flux, itp_mapS; warm=600.0, div_thresh=1e4)
    try
        N  = traj.N
        # attitude-only TL features (map deliberately excluded — see paper_impl.jl)
        x  = create_TL_A(flux; terms=TLNN_TERMS)
        Nf = size(x,2)
        (_,_,x_norm) = norm_sets(x)
        x_norm = Float32.(x_norm)

        # data-driven cold-start DC init from the first 5 min (onboard info only)
        Mw   = min(N, round(Int, 300/traj.dt))
        xz   = zeros(18, Mw)
        pred = MagNav.get_h(itp_mapS, xz, ins.lat[1:Mw], ins.lon[1:Mw], ins.alt[1:Mw]; core=true)
        intf = mag_uc[1:Mw] .- pred
        y_norms = (Float32(median(intf)), Float32(std(intf)))

        m     = MagNav.get_nn_m(Nf,1; hidden=TLNN_HIDDEN)
        nx_nn = length(MagNav.destructure(m)[1])
        P0_nn = Matrix(Diagonal(fill(TLNN_P0_SIGMA^2, nx_nn)))
        nnsig = fill(TLNN_WEIGHT_Q, nx_nn)

        (P0,Qd,R) = create_model(traj.dt,traj.lat[1];
                                 init_pos_sigma=0.1,init_alt_sigma=1.0,init_vel_sigma=1.0,
                                 meas_var=TLNN_MEAS_VAR,fogm_sigma=TLNN_FOGM_SIG,
                                 fogm_tau=TLNN_FOGM_TAU,vec_states=false,
                                 TL_sigma=nnsig,P0_TL=P0_nn)

        frw = ekf_online_nn(ins,mag_uc,itp_mapS,x_norm,m,y_norms,P0,Qd,R;
                            fogm_tau=TLNN_FOGM_TAU,core=true)
        fo  = MagNav.eval_filt(traj,ins,frw)
        m0  = round(Int, warm/traj.dt) + 1
        m0 >= N && (m0 = 1)
        ind = m0:N
        d   = sqrt(mean(dlat2dn.(fo.lat[ind].-traj.lat[ind],traj.lat[ind]).^2 .+
                        dlon2de.(fo.lon[ind].-traj.lon[ind],traj.lat[ind]).^2))
        return (isfinite(d) && d <= div_thresh) ? d : Inf
    catch e
        @warn("ekf_tlnn_drms failed", e)
        return Inf
    end
end
