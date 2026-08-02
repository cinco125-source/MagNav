#!/usr/bin/env python3
"""Python port of export_line.jl for the FULL line 1007.06 (no Julia needed).

Rebuilds every ingredient the GTSAM scripts consume — Pinson Phi tensor,
Tolles-Lawson A, INS/truth positions, (map+IGRF) grid — directly from the raw
SGL 2020 flight HDF5 and the Renfrew_395 map, mirroring src/get_XYZ.jl,
src/model_functions.jl (get_pinson/get_Phi), src/tolles_lawson.jl
(create_TL_A, central fdm), and research/fgo_breadth.jl parameters
(win=300, overlap=90, Huber, MEAS_VAR=12^2, FOGM 3/180).

Self-check: the first 6000 samples must reproduce research/gtsam_poc/
line_1007_06.h5 (which Julia exported for the same line) to float32 accuracy.

Usage:
  export_full_line.py <Flt1007_train.h5> <Renfrew_395_map.h5> <out.h5> \
      [--line 1003.02] [--validate line_1007_06.h5] [--comp] [--no-phi]

--comp adds the aeromagnetic-compensation channels the navigation export never
pulled: every scalar magnetometer including the mag_1_c stinger reference, all
four fluxgates in raw x/y/z/t rather than only the Tolles-Lawson matrix built
from flux D, the 14 current channels, the voltage rails with vol_cabt, and the
auxiliary series. --no-phi drops the Pinson tensor, which is the bulk of the
file and is of no use to a compensation study.

  # navigation (unchanged)
  export_full_line.py Flt1007_train.h5 Renfrew_395.h5 line_1007_06_full.h5
  # compensation, small file
  export_full_line.py Flt1007_train.h5 Renfrew_395.h5 comp_1007_06.h5 \
      --comp --no-phi
"""
import sys
import numpy as np
import h5py
from scipy.linalg import expm

R_EARTH  = 6378137.0
W_EARTH  = 7.2921151467e-5
G_EARTH  = 9.80665

# df_nav (examples/dataframes/df_nav.csv) line windows; overridable via CLI
LINES = {
    "1003.02": (50713.0, 54497.0, 1003.02),
    "1003.08": (60243.0, 64586.0, 1003.08),
    "1006.08": (55770.0, 56609.0, 1006.08),
    "1007.02": (48024.0, 51880.0, 1007.02),
    "1007.06": (57770.0, 63010.0, 1007.06),
}
T_START, T_END, LINE = LINES["1007.06"]
FOGM_TAU = 180.0
TAU      = 3600.0                                   # baro/acc/gyro tau
DT_EXP   = 0.1
DATE     = 2020 + (185 - 0.5) / 366.0               # get_years(2020,185), 2020 is leap


def euler2dcm(roll, pitch, yaw):
    """body2nav DCM, Titterton & Weston p.41 (mirrors src/dcm.jl)."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    dcm = np.zeros((3, 3) + np.shape(roll))
    dcm[0, 0] = cp*cy
    dcm[0, 1] = -cr*sy + sr*sp*cy
    dcm[0, 2] = sr*sy + cr*sp*cy
    dcm[1, 0] = cp*sy
    dcm[1, 1] = cr*cy + sr*sp*sy
    dcm[1, 2] = -sr*cy + cr*sp*sy
    dcm[2, 0] = -sp
    dcm[2, 1] = sr*cp
    dcm[2, 2] = cr*cp
    return dcm


def fdm(x):
    """central finite difference, mirrors src/tolles_lawson.jl fdm."""
    x = np.asarray(x, dtype=float)
    d = np.empty_like(x)
    d[0] = x[1] - x[0]
    d[-1] = x[-1] - x[-2]
    d[1:-1] = (x[2:] - x[:-2]) / 2
    return d


def create_TL_A(Bx, By, Bz, Bt_scale=50000.0):
    """terms = [:permanent,:induced,:eddy,:bias] -> 19 columns."""
    Bt = np.sqrt(Bx**2 + By**2 + Bz**2)
    bxh, byh, bzh = Bx/Bt, By/Bt, Bz/Bt
    bxd, byd, bzd = fdm(Bx), fdm(By), fdm(Bz)
    s = Bt_scale
    cols = [bxh, byh, bzh,
            bxh*Bx/s, bxh*By/s, bxh*Bz/s, byh*By/s, byh*Bz/s, bzh*Bz/s,
            bxh*bxd/s, bxh*byd/s, bxh*bzd/s,
            byh*bxd/s, byh*byd/s, byh*bzd/s,
            bzh*bxd/s, bzh*byd/s, bzh*bzd/s,
            np.ones_like(Bt)]
    return np.column_stack(cols)


def get_pinson(nx, lat, vn, ve, vd, fn, fe, fd, Cnb,
               k1=3e-2, k2=3e-4, k3=1e-6, fogm_tau=FOGM_TAU, tau=TAU):
    """mirrors src/model_functions.jl get_pinson (fogm_state=true)."""
    tl, cl, sl = np.tan(lat), np.cos(lat), np.sin(lat)
    r = R_EARTH
    F = np.zeros((nx, nx))
    F[0, 2] = -vn / r**2
    F[0, 3] = 1 / r
    F[1, 0] = ve * tl / (r * cl)
    F[1, 2] = -ve / (cl * r**2)
    F[1, 4] = 1 / (r * cl)
    F[2, 2] = -k1
    F[2, 5] = -1
    F[2, 9] = k1
    F[3, 0] = -ve * (2*W_EARTH*cl + ve / (r * cl**2))
    F[3, 2] = (ve**2 * tl - vn*vd) / r**2
    F[3, 3] = vd / r
    F[3, 4] = -2*(W_EARTH*sl + ve*tl / r)
    F[3, 5] = vn / r
    F[3, 7] = -fd
    F[3, 8] = fe
    F[4, 0] = 2*W_EARTH*(vn*cl - vd*sl) + vn*ve / (r * cl**2)
    F[4, 2] = -ve*((vn*tl + vd) / r**2)
    F[4, 3] = 2*W_EARTH*sl + ve*tl / r
    F[4, 4] = (vn*tl + vd) / r
    F[4, 5] = 2*W_EARTH*cl + ve / r
    F[4, 6] = fd
    F[4, 8] = -fn
    F[5, 0] = 2*W_EARTH*ve*sl
    F[5, 2] = (vn**2 + ve**2) / r**2 + k2
    F[5, 3] = -2*vn / r
    F[5, 4] = -2*(W_EARTH*cl + ve / r)
    F[5, 6] = -fe
    F[5, 7] = fn
    F[5, 9] = -k2
    F[5, 10] = 1
    F[6, 0] = -W_EARTH*sl
    F[6, 2] = -ve**2 / r**2
    F[6, 4] = 1 / r
    F[6, 7] = -W_EARTH*sl - ve*tl / r
    F[6, 8] = vn / r
    F[7, 2] = vn / r**2
    F[7, 3] = -1 / r
    F[7, 6] = W_EARTH*sl + ve*tl / r
    F[7, 8] = W_EARTH*cl + ve / r
    F[8, 0] = -W_EARTH*cl - ve / (r * cl**2)
    F[8, 2] = ve*tl / r**2
    F[8, 4] = -tl / r
    F[8, 6] = -vn / r
    F[8, 7] = -W_EARTH*cl - ve / r
    F[9, 9] = -1 / tau
    F[10, 2] = k3
    F[10, 9] = -k3
    for i in range(11, 17):
        F[i, i] = -1 / tau
    F[3:6, 11:14] = Cnb
    F[6:9, 14:17] = -Cnb
    F[nx-1, nx-1] = -1 / fogm_tau
    return F


# --comp: the channels an aeromagnetic-compensation study needs and the
# navigation export never pulled. All of them are in the SGL flight file; the
# gap was here, not in the data. Without mag_1_c there is no clean reference to
# score a compensated cabin sensor against, which is the metric a compensation
# paper is built on, and the export carried only the Tolles-Lawson matrix built
# from flux D, so neither the raw fluxgates nor the other three booms survived.
# Fields absent from a given flight are skipped and reported, since not every
# SGL flight records every channel.
COMP_SCALAR = ["mag_1_c", "mag_1_uc", "mag_2_c", "mag_2_uc", "mag_3_c",
               "mag_3_uc", "mag_4_uc", "mag_5_uc", "mag_6_uc"]
COMP_FLUX = [f"flux_{b}_{c}" for b in "abcd" for c in "xyzt"]
COMP_CUR = ["cur_com_1", "cur_ac_hi", "cur_ac_lo", "cur_tank", "cur_flap",
            "cur_strb", "cur_srvo_o", "cur_srvo_m", "cur_srvo_i", "cur_heat",
            "cur_acpwr", "cur_outpwr", "cur_bat_1", "cur_bat_2"]
# vol_cabt is the cabin temperature channel; the rest are the platform's
# voltage rails, the other half of the electrical-interference picture
COMP_VOL = ["vol_acc_p", "vol_acc_n", "vol_block", "vol_back", "vol_back_p",
            "vol_back_n", "vol_srvo", "vol_cabt", "vol_fan", "vol_bat_1",
            "vol_bat_2", "vol_res_p", "vol_res_n", "vol_outpwr", "vol_acpwr",
            "vol_gyro_1", "vol_gyro_2"]
COMP_AUX = ["diurnal", "igrf", "ogs_mag", "ogs_alt", "utm_x", "utm_y", "utm_z",
            "baro", "radar", "topo", "dem", "drape"]


def read_comp(f, ind):
    """Read every compensation channel present in the flight file."""
    out, missing = {}, []
    for k in COMP_SCALAR + COMP_FLUX + COMP_CUR + COMP_VOL + COMP_AUX:
        if k in f:
            out[k] = np.asarray(f[k][()])[ind].astype(float)
        else:
            missing.append(k)
    return out, missing


def main():
    flt_path, map_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
    val_path = sys.argv[sys.argv.index("--validate") + 1] if "--validate" in sys.argv else None
    if "--line" in sys.argv:
        t_start, t_end, line_no = LINES[sys.argv[sys.argv.index("--line") + 1]]
    else:
        t_start, t_end, line_no = T_START, T_END, LINE

    f = h5py.File(flt_path, "r")
    tt = f["tt"][()]
    line = f["line"][()]
    ind = (tt >= t_start) & (tt <= t_end) & (np.round(line, 2) == line_no)
    N = int(ind.sum())
    dt = float(f["dt"][()])
    print(f"line {line_no}: N={N} ({N*dt/60:.1f} min)", flush=True)

    # truth (deg -> rad) and INS (already rad; mirrors get_XYZ20)
    true_lat = np.deg2rad(f["lat"][()][ind]); true_lon = np.deg2rad(f["lon"][()][ind])
    ins_lat = f["ins_lat"][()][ind].astype(float)
    ins_lon = f["ins_lon"][()][ind].astype(float)
    ins_alt = f["ins_alt"][()][ind].astype(float)
    vn = f["ins_vn"][()][ind].astype(float)
    ve = -f["ins_vw"][()][ind].astype(float)
    vd = -f["ins_vu"][()][ind].astype(float)
    wander = f["ins_wander"][()][ind].astype(float)
    roll  = np.deg2rad(f["ins_roll"][()][ind])
    pitch = np.deg2rad(f["ins_pitch"][()][ind])
    yaw   = np.deg2rad(f["ins_yaw"][()][ind])
    ax = f["ins_acc_x"][()][ind].astype(float)
    ay = f["ins_acc_y"][()][ind].astype(float)
    az = f["ins_acc_z"][()][ind].astype(float)
    Bx = f["flux_d_x"][()][ind].astype(float)
    By = f["flux_d_y"][()][ind].astype(float)
    Bz = f["flux_d_z"][()][ind].astype(float)
    meas = f["mag_5_uc"][()][ind].astype(float)
    meas4 = f["mag_4_uc"][()][ind].astype(float)
    comp, comp_missing = ({}, [])
    if "--comp" in sys.argv:
        comp, comp_missing = read_comp(f, ind)
        print(f"comp channels: {len(comp)} present, {len(comp_missing)} absent",
              flush=True)
        if comp_missing:
            print("  absent: " + " ".join(comp_missing), flush=True)
        for k in ("mag_1_c", "flux_a_x"):
            if k not in comp:
                print(f"  WARNING: {k} absent -- this flight cannot support the "
                      f"compensation reference/basis that depends on it", flush=True)
    f.close()

    # specific force: rotate wander (CW for NED), mirrors get_XYZ20
    cw, sw = np.cos(-wander), np.sin(-wander)
    fx, fy, fz = ax, -ay, -az
    fn = cw*fx - sw*fy
    fe = sw*fx + cw*fy
    fd = fz

    Cnb = euler2dcm(roll, pitch, yaw)      # (3,3,N)

    # zero_ins_ll N_zero=1: remove the initial INS-truth offset everywhere
    ins_lat = ins_lat - (ins_lat[0] - true_lat[0])
    ins_lon = ins_lon - (ins_lon[0] - true_lon[0])

    A = create_TL_A(Bx, By, Bz)
    nTL = A.shape[1]

    # P0/Qd/R from the Julia segment export: same create_model params and the
    # SAME lat[1] (the segment is the first 10 min of this very line)
    with h5py.File("research/gtsam_poc/line_1007_06.h5", "r") as g:
        P0 = np.asarray(g["P0"], dtype=float)
        Qd = np.asarray(g["Qd"], dtype=float)
        Rm = float(g["R"][()])
    nx = P0.shape[0]

    # --no-phi: the Pinson tensor is (N-1, nx, nx) float32, about 570 MB raw on a
    # full 87-minute line and the bulk of the output. A compensation study never
    # touches it, so skip it and the file drops to the channels themselves.
    skip_phi = "--no-phi" in sys.argv
    Phi = None
    if skip_phi:
        print("skipping Phi tensor (--no-phi)", flush=True)
    else:
        print("building Phi tensor...", flush=True)
        Phi = np.zeros((N-1, nx, nx), dtype=np.float32)
        for t in range(N-1):
            F = get_pinson(nx, ins_lat[t], vn[t], ve[t], vd[t],
                           fn[t], fe[t], fd[t], Cnb[:, :, t])
            Phi[t] = expm(F * dt).astype(np.float32)
            if t % 10000 == 0:
                print(f"  Phi {t}/{N-1}", flush=True)

    # ---- map + IGRF grid ----
    print("building map grid...", flush=True)
    import ppigrf
    from scipy.interpolate import RectBivariateSpline
    with h5py.File(map_path, "r") as m:
        mmap = np.asarray(m["map"], dtype=float)
        mxx = np.asarray(m["xx"]).ravel()     # lon [rad]? (validated below)
        myy = np.asarray(m["yy"]).ravel()     # lat [rad]?
        malt_map = float(np.asarray(m["alt"]).ravel()[0])
    # normalize orientation: map[i,j] should be (yy[i], xx[j])
    if mmap.shape == (mxx.size, myy.size) and mxx.size != myy.size:
        mmap = mmap.T
    if mmap.shape != (myy.size, mxx.size):
        mmap = mmap.T
    assert mmap.shape == (myy.size, mxx.size)
    if np.max(np.abs(myy)) > np.pi:           # deg -> rad if needed
        myy = np.deg2rad(myy); mxx = np.deg2rad(mxx)

    malt = float(np.mean(ins_alt))
    dz = malt - malt_map
    if dz > 20.0:
        # upward continuation to flight altitude (mirrors Julia upward_fft):
        # FFT, multiply by e^{-k dz}, IFFT. Skipped when dz is a few metres
        # (validated harmless for 1007.06); essential for the ~400 m gap of
        # 1007.02 over Eastern_395.
        ny_, nx_ = mmap.shape
        dy_m = (myy[1] - myy[0]) * R_EARTH
        dx_m = (mxx[1] - mxx[0]) * R_EARTH * np.cos(myy.mean())
        # mirror-pad to soften wrap-around
        py_, px_ = ny_ // 4, nx_ // 4
        big = np.pad(mmap, ((py_, py_), (px_, px_)), mode="reflect")
        ky = np.fft.fftfreq(big.shape[0], d=dy_m) * 2 * np.pi
        kx = np.fft.fftfreq(big.shape[1], d=dx_m) * 2 * np.pi
        K = np.sqrt(ky[:, None]**2 + kx[None, :]**2)
        big = np.real(np.fft.ifft2(np.fft.fft2(big) * np.exp(-K * dz)))
        mmap = big[py_:py_+ny_, px_:px_+nx_]
        print(f"upward-continued map {dz:.0f} m to flight altitude", flush=True)
    itp = RectBivariateSpline(myy, mxx, mmap, kx=3, ky=3, s=0)
    pad = 0.02 * np.pi / 180
    # grid density: 113 m spacing (Ng=600) was enough for Mag 5 but is a prime
    # suspect for the Mag 4 hard-regime divergence (gradient fidelity)
    Ng = int(sys.argv[sys.argv.index("--ng") + 1]) if "--ng" in sys.argv else 600
    glat = np.linspace(true_lat.min()-pad, true_lat.max()+pad, Ng)
    glon = np.linspace(true_lon.min()-pad, true_lon.max()+pad, Ng)
    gmap = itp(glat, glon)                     # (Ng, Ng) map anomaly [nT]
    # IGRF magnitude on the grid (geodetic), date 2020.5
    from datetime import datetime, timedelta
    date_dt = datetime(2020, 1, 1) + timedelta(days=184.5)
    LON, LAT = np.meshgrid(np.rad2deg(glon), np.rad2deg(glat))
    # ppigrf allocates an (npoints, nterms) Legendre table, so a single call on a
    # fine grid (Ng=2000 -> 4e6 points) exhausts memory. Evaluate row-by-row in
    # chunks; the result is bit-identical to the one-shot call.
    gigrf = np.empty((Ng, Ng))
    rows_per_chunk = max(1, 250_000 // Ng)
    for i0 in range(0, Ng, rows_per_chunk):
        i1 = min(i0 + rows_per_chunk, Ng)
        Be, Bn, Bu = ppigrf.igrf(LON[i0:i1], LAT[i0:i1], malt/1000.0, date_dt)
        chunk = np.sqrt(Be**2 + Bn**2 + Bu**2)[0 if np.ndim(Be) == 3 else ...]
        gigrf[i0:i1] = np.squeeze(chunk).reshape(i1 - i0, Ng)
    gh = gmap + gigrf
    print(f"grid {Ng}x{Ng}, malt={malt:.1f} m, map alt={malt_map:.1f} m, "
          f"spacing ~{(glat[1]-glat[0])*R_EARTH:.0f} m", flush=True)

    # ---- validation against the Julia 10-min export ----
    if val_path:
        with h5py.File(val_path, "r") as g:
            Nv = int(g["N"][()])
            vPhi = np.asarray(g["Phi"], dtype=float)
            if vPhi.shape[0] == nx:
                vPhi = np.moveaxis(vPhi, 2, 0)
            else:
                vPhi = vPhi.transpose(0, 2, 1)
            vA = np.asarray(g["A"], dtype=float)
            if vA.shape[0] != Nv: vA = vA.T
            vmeas = np.asarray(g["meas"]).ravel()
            vins_lat = np.asarray(g["ins_lat"]).ravel()
            vins_lon = np.asarray(g["ins_lon"]).ravel()
            vtrue_lat = np.asarray(g["true_lat"]).ravel()
            vglat = np.asarray(g["glat"]).ravel()
            vglon = np.asarray(g["glon"]).ravel()
            vgh = np.asarray(g["gh"], dtype=float).T
            vmalt = float(g["malt"][()])
        print("=== validation vs Julia segment export ===", flush=True)
        print(f"meas   max|diff| : {np.abs(meas[:Nv]-vmeas).max():.3e} nT")
        print(f"insLat max|diff| : {np.abs(ins_lat[:Nv]-vins_lat).max()*R_EARTH:.3e} m")
        print(f"insLon max|diff| : {np.abs(ins_lon[:Nv]-vins_lon).max()*R_EARTH:.3e} m")
        print(f"truLat max|diff| : {np.abs(true_lat[:Nv]-vtrue_lat).max()*R_EARTH:.3e} m")
        print(f"A      max|diff| : {np.abs(A[:Nv]-vA).max():.3e}")
        dP = np.abs(Phi[:Nv-1].astype(float) - vPhi[:Nv-1])
        print(f"Phi    max|diff| : {dP.max():.3e}  (float32 eps ~1e-7 rel)")
        # map grid: evaluate our itp+igrf on the Julia grid and compare
        vg_map = itp(vglat, vglon)
        LONv, LATv = np.meshgrid(np.rad2deg(vglon), np.rad2deg(vglat))
        Bev, Bnv, Buv = ppigrf.igrf(LONv, LATv, vmalt/1000.0, date_dt)
        vg = vg_map + np.squeeze(np.sqrt(Bev**2 + Bnv**2 + Buv**2))
        dmap = vg - vgh
        print(f"grid   diff mean/std/max: {dmap.mean():.2f} / {dmap.std():.2f} / "
              f"{np.abs(dmap).max():.2f} nT", flush=True)

    with h5py.File(out_path, "w") as o:
        o["N"] = N; o["dt"] = dt; o["nx"] = nx; o["nTL"] = nTL
        if Phi is not None:
            o.create_dataset("Phi", data=Phi, compression="gzip", compression_opts=1)
        o["A"] = A; o["meas"] = meas               # mag_5_uc (back-compat)
        o["meas_mag4"] = meas4; o["meas_mag5"] = meas
        o["ins_lat"] = ins_lat; o["ins_lon"] = ins_lon; o["ins_alt"] = ins_alt
        o["true_lat"] = true_lat; o["true_lon"] = true_lon
        o["P0"] = P0; o["Qd"] = Qd; o["R"] = Rm
        o["glat"] = glat; o["glon"] = glon
        o["gh_rowmajor"] = gh                  # ALREADY (lat, lon) row-major
        o["malt"] = malt
        o["warm"] = 600.0; o["win"] = 300.0; o["overlap"] = 90.0
        o["ref_drms"] = float("nan")
        o["tt"] = tt[ind].astype(float)
        o["ins_roll"] = roll; o["ins_pitch"] = pitch; o["ins_yaw"] = yaw
        for k, v in comp.items():
            o.create_dataset(k, data=v, compression="gzip", compression_opts=1)
        if comp:
            print(f"wrote {len(comp)} compensation channels", flush=True)
    print("wrote", out_path, flush=True)

if __name__ == "__main__":
    main()
