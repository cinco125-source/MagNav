#!/usr/bin/env python3
"""What does a magnetic-anomaly map actually observe, and does a turn change it?

The question came out of asking whether the map could calibrate the IMU. On the
recorded 1007.06 segment -- ten minutes of straight and level -- it cannot: the
accelerometer bias posterior is 0.992 of its prior, 0.1 dB, and the gyro bias
posterior is WORSE than its prior because process noise grows it faster than a
scalar measurement shrinks it. Velocity gets 12 to 18 dB and horizontal tilt 8
to 10, so the map is informative; it is informative about the wrong states.

The reason is structural rather than numerical. In the Pinson model the bias
enters through

    F[4:6, 12:14] =  Cnb        accelerometer bias, body frame -> NED velocity
    F[7:9, 15:17] = -Cnb        gyro bias,          body frame -> NED tilt

so a CONSTANT body-frame bias projects into NED along a direction fixed by the
attitude. Hold the attitude constant and that direction never moves, which
makes the bias indistinguishable from the tilt state it shares a column with.
Change the heading and the projection sweeps, and the two separate. Turning is
not a way to get more measurements; it is the only way to make these particular
states identifiable at all.

Every SGL flight LINE is straight -- the turns happen between lines, outside
the exported windows -- so the recorded data cannot answer this and a synthetic
trajectory has to. What is synthesized is only the trajectory: the map is the
real Renfrew/Eastern grid carried in the export, the gradients are read from
it, and the Tolles-Lawson regressor is built from the real Earth field rotated
through the synthetic attitude, so the compensation competes for the same
measurement it competes for in flight.

VALIDATION GATE. --validate runs the same covariance recursion on the REAL
exported Phi and A from the segment and must reproduce the straight-line
numbers above. If it does not, nothing below it means anything.

Usage:
  observability_audit.py <line.h5> --validate
  observability_audit.py <line.h5> [--bank 0,15,30,45] [--minutes 10] [--no-tl]
  observability_audit.py <line.h5> --gramian

WHAT IT FOUND. The posterior recursion says a turn does nothing for the bias:
ten minutes of 30-degree bank sweeps 2159 degrees of heading, drives velocity
from 1.8 to 25.5 dB and tilt from 2.2 to 20.5, and leaves the accelerometer
bias at 0.0 dB and the gyro bias at -2.9. That reading is misleading, because
the SGL prior is so tight that nothing the map says can move it -- P0 assumes
25 ug and 0.0015 deg/hr, which is navigation grade or better.

--gramian drops the prior and asks what the MAP ALONE would know, inverting the
observability Gramian of the propagated map row. The answer at t = 0:

  trajectory          accel bx        vs prior     gyro bx        vs prior
  straight   600 s    1.6e4 m/s2      6.6e7        0.074 rad/s    1e7
  turn 30    600 s    1811            7.4e6        10.7           1.5e9
  turn 30   3600 s      16.0          6.5e4        0.096          1.3e7
  racetrack 3600 s       0.0028       11           1.0e-6         140

The lever is not the turn, it is the RACETRACK. A continuous turn stays in one
place and covers no map; alternating straight legs with turns translates and
rotates at once, which is what separates a body-frame bias from an NED-frame
tilt while still sweeping map gradient. Same duration, six thousand times the
information.

In instrument units the racetrack hour gives 286 ug and 0.21 deg/hr from the
map alone. Against the navigation-grade INS SGL actually flies, that loses by
two orders. Against a tactical unit (1 to 10 deg/hr, 100 to 500 ug) it is
comparable to five times better, and against MEMS (10 to 100 deg/hr, 1 to
10 mg) it wins by one to two orders. Which is the result: a magnetic anomaly
map can calibrate a cheap IMU in flight, provided the trajectory turns, and the
reason nobody has seen it is that the public benchmark flies a navigation-grade
INS down straight lines -- the two worst conditions simultaneously.

These are linear observability figures for a synthetic trajectory over the real
map. They bound what is worth attempting; they do not demonstrate it.
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "gtsam_poc"))
from run_gtsam_decimated import Grid, load          # noqa: E402

R_EARTH = 6378137.0
OMEGA_E = 7.2921151467e-5
G0 = 9.80665
NAV = {0: "lat", 1: "lon", 2: "alt", 3: "vn", 4: "ve", 5: "vd",
       6: "tilt_n", 7: "tilt_e", 8: "tilt_d", 9: "baro", 10: "baro2",
       11: "accel bx", 12: "accel by", 13: "accel bz",
       14: "gyro bx", 15: "gyro by", 16: "gyro bz"}


def arg(flag, default, cast=str):
    return cast(sys.argv[sys.argv.index(flag) + 1]) if flag in sys.argv else default


def pinson_F(nx, lat, vn, ve, vd, fn, fe, fd, Cnb, baro_tau=3600.0,
             acc_tau=3600.0, gyro_tau=3600.0, fogm_tau=180.0,
             k1=3e-2, k2=3e-4, k3=1e-6):
    """Port of get_pinson (src/model_functions.jl:352), 1-indexed rows shifted."""
    tan_l, cos_l, sin_l = math.tan(lat), math.cos(lat), math.sin(lat)
    F = np.zeros((nx, nx))
    F[0, 2] = -vn / R_EARTH ** 2
    F[0, 3] = 1 / R_EARTH
    F[1, 0] = ve * tan_l / (R_EARTH * cos_l)
    F[1, 2] = -ve / (cos_l * R_EARTH ** 2)
    F[1, 4] = 1 / (R_EARTH * cos_l)
    F[2, 2] = -k1
    F[2, 5] = -1
    F[2, 9] = k1
    F[3, 0] = -ve * (2 * OMEGA_E * cos_l + ve / (R_EARTH * cos_l ** 2))
    F[3, 2] = (ve ** 2 * tan_l - vn * vd) / R_EARTH ** 2
    F[3, 3] = vd / R_EARTH
    F[3, 4] = -2 * (OMEGA_E * sin_l + ve * tan_l / R_EARTH)
    F[3, 5] = vn / R_EARTH
    F[3, 7] = -fd
    F[3, 8] = fe
    F[4, 0] = 2 * OMEGA_E * (vn * cos_l - vd * sin_l) + vn * ve / (R_EARTH * cos_l ** 2)
    F[4, 2] = -ve * ((vn * tan_l + vd) / R_EARTH ** 2)
    F[4, 3] = 2 * OMEGA_E * sin_l + ve * tan_l / R_EARTH
    F[4, 4] = (vn * tan_l + vd) / R_EARTH
    F[4, 5] = 2 * OMEGA_E * cos_l + ve / R_EARTH
    F[4, 6] = fd
    F[4, 8] = -fn
    F[5, 0] = 2 * OMEGA_E * ve * sin_l
    F[5, 2] = (vn ** 2 + ve ** 2) / R_EARTH ** 2 + k2
    F[5, 3] = -2 * vn / R_EARTH
    F[5, 4] = -2 * (OMEGA_E * cos_l + ve / R_EARTH)
    F[5, 6] = -fe
    F[5, 7] = fn
    F[5, 9] = -k2
    F[5, 10] = 1
    F[6, 0] = -OMEGA_E * sin_l
    F[6, 2] = -ve ** 2 / R_EARTH ** 2
    F[6, 4] = 1 / R_EARTH
    F[6, 7] = -OMEGA_E * sin_l - ve * tan_l / R_EARTH
    F[6, 8] = vn / R_EARTH
    F[7, 2] = vn / R_EARTH ** 2
    F[7, 3] = -1 / R_EARTH
    F[7, 6] = OMEGA_E * sin_l + ve * tan_l / R_EARTH
    F[7, 8] = OMEGA_E * cos_l + ve / R_EARTH
    F[8, 0] = -OMEGA_E * cos_l - ve / (R_EARTH * cos_l ** 2)
    F[8, 2] = ve * tan_l / R_EARTH ** 2
    F[8, 4] = -tan_l / R_EARTH
    F[8, 6] = -vn / R_EARTH
    F[8, 7] = -OMEGA_E * cos_l - ve / R_EARTH
    F[9, 9] = -1 / baro_tau
    F[10, 2] = k3
    F[10, 9] = -k3
    for i in (11, 12, 13):
        F[i, i] = -1 / acc_tau
    for i in (14, 15, 16):
        F[i, i] = -1 / gyro_tau
    F[3:6, 11:14] = Cnb
    F[6:9, 14:17] = -Cnb
    F[nx - 1, nx - 1] = -1 / fogm_tau
    return F


def dcm(roll, pitch, yaw):
    """Body-to-NED direction cosine matrix, ZYX."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array([
        [cp * cy, sr * sp * cy - cr * sy, cr * sp * cy + sr * sy],
        [cp * sy, sr * sp * sy + cr * cy, cr * sp * sy - sr * cy],
        [-sp,     sr * cp,                cr * cp]])


def trajectory(kind, lat0, lon0, V, bank_deg, T, dt, alt=400.0):
    """Level flight at constant speed: straight, a steady turn, or a racetrack.

    A coordinated level turn at bank phi has turn rate g tan(phi) / V and
    specific force (V*omega toward the centre, 0 vertical) minus gravity, which
    is what the tilt-to-velocity coupling in the Pinson matrix reads.
    """
    n = int(round(T / dt))
    bank = math.radians(bank_deg)
    om = G0 * math.tan(bank) / V if bank_deg else 0.0
    lat = np.zeros(n); lon = np.zeros(n); psi = np.zeros(n)
    lat[0], lon[0] = lat0, lon0
    Cnb = np.zeros((n, 3, 3))
    f = np.zeros((n, 3))
    vel = np.zeros((n, 3))
    for i in range(n):
        if kind == "racetrack":
            # 2 min straight, 1 min turn, repeating -- the shape a survey flies
            phase = (i * dt) % 360.0
            turning = phase >= 240.0
        else:
            turning = kind == "turn"
        w = om if turning else 0.0
        b = bank if turning else 0.0
        if i:
            psi[i] = psi[i - 1] + w * dt
            vn_, ve_ = V * math.cos(psi[i]), V * math.sin(psi[i])
            lat[i] = lat[i - 1] + vn_ / R_EARTH * dt
            lon[i] = lon[i - 1] + ve_ / (R_EARTH * math.cos(lat[i - 1])) * dt
        vn_, ve_ = V * math.cos(psi[i]), V * math.sin(psi[i])
        vel[i] = (vn_, ve_, 0.0)
        # centripetal acceleration points 90 deg right of the velocity for w>0
        a_n, a_e = -V * w * math.sin(psi[i]), V * w * math.cos(psi[i])
        f[i] = (a_n, a_e, -G0)
        Cnb[i] = dcm(b, 0.0, psi[i])
    return lat, lon, np.full(n, alt), vel, f, Cnb, psi


def tl_basis(Cnb, B_ned, dt, Bt_scale=50000.0):
    """The 19-column Tolles-Lawson regressor: 3 permanent, 6 induced, 9 eddy, 1 bias.

    Built from the real Earth field rotated into the synthetic body frame, so
    the columns are excited by attitude exactly as they are in flight -- which
    is the point, since a turn excites the compensation and the bias at once.
    """
    n = Cnb.shape[0]
    Bb = np.einsum("nji,j->ni", Cnb, B_ned)          # Cnb^T B_ned
    Bt = np.linalg.norm(Bb, axis=1, keepdims=True)
    c = Bb / Bt
    cd = np.gradient(c, dt, axis=0)
    A = np.zeros((n, 19))
    A[:, 0:3] = c
    k = 3
    for i in range(3):
        for j in range(i, 3):
            A[:, k] = Bt[:, 0] / Bt_scale * c[:, i] * c[:, j]
            k += 1
    for i in range(3):
        for j in range(3):
            A[:, k] = Bt[:, 0] / Bt_scale * c[:, i] * cd[:, j]
            k += 1
    A[:, 18] = 1.0
    return A


def recurse(Phi, Qd, P0, Rm, H_map, A, nx, iTL, iS, use_tl):
    """Covariance-only recursion. No measurements needed -- information does not
    depend on their values, only on the geometry."""
    P = P0.copy()
    M = Phi.shape[0] + 1
    for s in range(M):
        if s:
            P = Phi[s - 1] @ P @ Phi[s - 1].T + Qd
            P = (P + P.T) / 2
        H = np.zeros((1, nx))
        H[0, 0], H[0, 1] = H_map[s]
        if use_tl:
            H[0, iTL] = A[s]
        H[0, iS] = 1.0
        S = float((H @ P @ H.T)[0, 0]) + Rm
        K = (P @ H.T / S).ravel()
        KH = np.eye(nx) - np.outer(K, H.ravel())
        P = KH @ P @ KH.T + np.outer(K, K) * Rm
        P = (P + P.T) / 2
    return P


def report(title, sig0, sigN, extra=""):
    print(f"\n=== {title} {extra}")
    print(f"{'state':10s}{'prior sd':>12s}{'post sd':>12s}{'shrink':>9s}{'gain':>10s}")
    for i, n in NAV.items():
        r = sigN[i] / sig0[i]
        print(f"{n:10s}{sig0[i]:12.4g}{sigN[i]:12.4g}{r:9.3f}"
              f"{-20*math.log10(max(r,1e-12)):8.1f} dB")


def gramian_mode(d, grid, nx, nTL, Rm, P0, iTL, iS):
    """What the map alone would know, with every state free and no prior."""
    lat0 = float(np.mean(d["glat"])); lon0 = float(np.mean(d["glon"]))
    inc = math.radians(72.0)
    B = 54000.0 * np.array([math.cos(inc), 0.0, math.sin(inc)])
    sig0 = np.sqrt(np.diag(P0))
    print("map-only standard deviation at t=0, prior discarded entirely")
    print(f"{'trajectory':22s}{'swept':>8s}{'accel bx':>12s}{'/prior':>10s}"
          f"{'gyro bx':>12s}{'/prior':>10s}{'vn':>10s}{'/prior':>9s}")
    for kind, bank, T in (("straight", 0, 600), ("turn", 30, 600),
                          ("turn", 30, 3600), ("racetrack", 30, 3600)):
        lat, lon, _, vel, f, Cnb, psi = trajectory(kind, lat0, lon0, 90.0,
                                                   bank, T, 1.0)
        A = tl_basis(Cnb, B, 1.0)
        Wo = np.zeros((nx, nx)); Psi = np.eye(nx)
        for s in range(lat.size):
            if s:
                F = pinson_F(nx, lat[s-1], vel[s-1, 0], vel[s-1, 1], vel[s-1, 2],
                             f[s-1, 0], f[s-1, 1], f[s-1, 2], Cnb[s-1])
                Psi = (np.eye(nx) + F + 0.5 * F @ F) @ Psi
            gl, go = grid.grad(lat[s], lon[s])
            H = np.zeros(nx); H[0] = gl; H[1] = go; H[iTL] = A[s]; H[iS] = 1.0
            h = H @ Psi
            Wo += np.outer(h, h) / Rm
        sd = np.sqrt(np.abs(np.diag(np.linalg.inv(Wo + 1e-30 * np.eye(nx)))))
        print(f"{kind+' '+str(int(bank))+' '+str(int(T))+'s':22s}"
              f"{math.degrees(psi[-1]-psi[0]):8.0f}"
              f"{sd[11]:12.4g}{sd[11]/sig0[11]:10.2g}"
              f"{sd[14]:12.4g}{sd[14]/sig0[14]:10.2g}"
              f"{sd[3]:10.4g}{sd[3]/sig0[3]:9.2g}")
    print()
    print("in instrument units, the racetrack hour: "
          f"accel {sd[11]/G0*1e6:.0f} ug, gyro {sd[14]*180/math.pi*3600:.2f} deg/hr")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") \
        else "/tmp/line_1007_06_py.h5"
    d = load(path)
    N = int(d["N"]); nx = int(d["nx"]); nTL = int(d["nTL"])
    dt = float(d["dt"]); Rm = float(d["R"])
    P0 = np.asarray(d["P0"], float)
    Qd = np.asarray(d["Qd"], float) + 1e-20 * np.eye(nx)
    gh = d["gh_rowmajor"] if "gh_rowmajor" in d else np.asarray(d["gh"])
    grid = Grid(d["glat"], d["glon"], gh)
    iTL = slice(17, 17 + nTL)
    iS = nx - 1
    P0 = P0.copy()
    P0[iTL, iTL] = P0[iTL, iTL] * 100.0 ** 2
    sig0 = np.sqrt(np.diag(P0))
    K = 10
    idx = np.arange(0, N, K)
    dtK = dt * K
    use_tl = "--no-tl" not in sys.argv

    if "--gramian" in sys.argv:
        gramian_mode(d, grid, nx, nTL, Rm, P0, iTL, iS)
        return

    if "--validate" in sys.argv:
        Phi = np.asarray(d["Phi"], float)
        A = np.asarray(d["A"], float)
        if A.shape[0] != N:
            A = A.T
        il = np.asarray(d["ins_lat"]).ravel()[idx]
        io = np.asarray(d["ins_lon"]).ravel()[idx]
        M = idx.size
        PhiK = np.zeros((M - 1, nx, nx)); QdK = np.zeros((M - 1, nx, nx))
        for s in range(M - 1):
            P_ = np.eye(nx); Q_ = np.zeros((nx, nx))
            for t in range(idx[s], min(idx[s + 1], N - 1)):
                P_ = Phi[t] @ P_
                Q_ = Phi[t] @ Q_ @ Phi[t].T + Qd
            PhiK[s] = P_; QdK[s] = (Q_ + Q_.T) / 2
        Hm = np.array([grid.grad(il[s], io[s]) for s in range(M)])
        P = recurse(PhiK, QdK[0] * 0 + QdK.mean(axis=0), P0, Rm, Hm,
                    A[idx], nx, iTL, iS, use_tl)
        report("VALIDATION: real exported Phi and A, recorded straight line",
               sig0, np.sqrt(np.diag(P)),
               f"({M} states, {M*dtK:.0f} s)")
        print("\naccel bias near 0 dB and gyro bias negative reproduces the")
        print("direct Kalman run, so the recursion is measuring what it claims.")
        return

    V = float(arg("--speed", 90.0, float))
    T = float(arg("--minutes", 10.0, float)) * 60.0
    banks = [float(x) for x in arg("--bank", "0,15,30,45").split(",")]
    lat0 = float(np.mean(d["glat"])); lon0 = float(np.mean(d["glon"]))
    # Earth field at the survey: ~54 uT, inclination ~72 deg for eastern Ontario
    inc = math.radians(72.0)
    B_ned = 54000.0 * np.array([math.cos(inc), 0.0, math.sin(inc)])
    print(f"synthetic trajectories over the REAL map, {T/60:g} min at {V:g} m/s, "
          f"states every {dtK:g} s, TL {'carried' if use_tl else 'absent'}")

    for kind in ("straight", "turn", "racetrack"):
        for bank in (banks if kind != "straight" else [0.0]):
            lat, lon, alt, vel, f, Cnb, psi = trajectory(
                kind, lat0, lon0, V, bank, T, dtK)
            M = lat.size
            A = tl_basis(Cnb, B_ned, dtK)
            PhiK = np.zeros((M - 1, nx, nx))
            for s in range(M - 1):
                F = pinson_F(nx, lat[s], vel[s, 0], vel[s, 1], vel[s, 2],
                             f[s, 0], f[s, 1], f[s, 2], Cnb[s])
                PhiK[s] = np.eye(nx) + F * dtK + 0.5 * (F * dtK) @ (F * dtK)
            Hm = np.array([grid.grad(lat[s], lon[s]) for s in range(M)])
            QdKm = Qd * K
            P = recurse(PhiK, QdKm, P0, Rm, Hm, A, nx, iTL, iS, use_tl)
            hdg = math.degrees(psi[-1] - psi[0])
            report(f"{kind}, bank {bank:g} deg", sig0, np.sqrt(np.diag(P)),
                   f"(heading swept {hdg:.0f} deg)")


if __name__ == "__main__":
    main()
