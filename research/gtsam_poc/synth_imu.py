#!/usr/bin/env python3
"""Synthesize an IMU from the recorded trajectory, and prove it before using it.

WHY. Our factor graph carries a Pinson error state with Phi precomputed outside
the graph, so the mechanization was already linearized before any factor was
built and the only nonlinearity left is a smooth map lookup. That is why the
frozen-linearization graph reproduces a Kalman filter to 8.5 nanometres and why
relinearization buys 1 to 3 percent. A tightly-coupled formulation -- attitude
on SO(3), raw inertial measurements preintegrated between keyframes, bias as a
first-class variable -- gives the graph a genuinely nonlinear problem. Whether
that changes the answer is the open question this file exists to set up.

WHAT IT IS NOT. SGL records ins_acc_* and *_rate, but those are the INS's own
compensated outputs, not raw increments, so preintegration cannot use them: the
bias we would be estimating has already been removed. The IMU here is therefore
synthesized from the recorded truth trajectory. That makes anything built on it
a MECHANISM STUDY, not a benchmark -- truth and inertial data share a source, so
the inertial error is exactly what we inject and no number from it belongs in a
table next to the recorded-INS results.

FRAME. A local NED frame at the start of the segment, treated as inertial.
Earth rate and transport rate are omitted, and the synthesized IMU is
consistent with that choice, so the graph and any filter compared against it
see the same physics. This is a simplification of navigation but not of the
question being asked, which is about linearization and not about the Earth.

THE GATE. Two checks must pass before this file is used for anything:
  1. round trip -- integrating the CLEAN synthetic IMU must reproduce the
     trajectory it was made from, to well under a metre over the segment.
  2. drift realism -- with injected errors it must reproduce the magnitude of
     the recorded INS drift, which is what makes the comparison meaningful.

Usage: synth_imu.py <line.h5> [--rate 100] [--gyro-bias 0.24] [--accel-bias 12]
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_gtsam_decimated import load                 # noqa: E402

R_EARTH = 6378137.0
G0 = 9.80665
DEG_PER_HR = math.pi / 180.0 / 3600.0
UG = 9.80665e-6


def arg(flag, default, cast=str):
    return cast(sys.argv[sys.argv.index(flag) + 1]) if flag in sys.argv else default


def ned_from_llh(lat, lon, alt, lat0, lon0, alt0):
    """Flat local NED about the segment start. Over 40 km the curvature error is
    metres in the vertical and centimetres horizontally, which is well inside
    the tolerance of a mechanism study."""
    n = (lat - lat0) * R_EARTH
    e = (lon - lon0) * R_EARTH * math.cos(lat0)
    d = -(alt - alt0)
    return np.stack([n, e, d], axis=-1)


def smooth(x, w):
    """Zero-phase moving average. Differentiating a 10 Hz track twice amplifies
    quantization badly, so the position is smoothed before the derivatives are
    taken; the window is reported so the reader can see what was removed."""
    if w <= 1:
        return x.copy()
    k = np.ones(w) / w
    pad = w // 2
    xp = np.pad(x, ((pad, pad), (0, 0)), mode="edge")
    return np.stack([np.convolve(xp[:, i], k, mode="same")[pad:-pad or None]
                     for i in range(x.shape[1])], axis=1)


def dcm_from_euler(roll, pitch, yaw):
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    R = np.empty(roll.shape + (3, 3))
    R[..., 0, 0] = cp * cy
    R[..., 0, 1] = sr * sp * cy - cr * sy
    R[..., 0, 2] = cr * sp * cy + sr * sy
    R[..., 1, 0] = cp * sy
    R[..., 1, 1] = sr * sp * sy + cr * cy
    R[..., 1, 2] = cr * sp * sy - sr * cy
    R[..., 2, 0] = -sp
    R[..., 2, 1] = sr * cp
    R[..., 2, 2] = cr * cp
    return R


def synthesize(pos_ned, dt, smooth_w=201):
    """Trajectory -> (attitude, body rates, specific force), coordinated flight.

    Heading follows the ground track, pitch follows the climb angle, and bank is
    the coordinated-turn angle implied by the turn rate and speed. That is an
    assumption about the aircraft, not a measurement, and it is the reason this
    IMU is synthetic: a real aircraft sideslips, and the recorded attitude would
    differ. It is consistent, which is what the mechanism study needs.
    """
    p = smooth(pos_ned, smooth_w)
    v = np.gradient(p, dt, axis=0)
    a = np.gradient(v, dt, axis=0)
    spd_h = np.hypot(v[:, 0], v[:, 1])
    yaw = np.arctan2(v[:, 1], v[:, 0])
    pitch = np.arctan2(-v[:, 2], np.maximum(spd_h, 1e-6))
    yaw_u = np.unwrap(yaw)
    turn = np.gradient(yaw_u, dt)
    roll = np.arctan2(turn * np.maximum(spd_h, 1e-6), G0)
    Rnb = dcm_from_euler(roll, pitch, yaw_u)
    # specific force in body: f = R^T (a - g),  g = (0,0,+G0) in NED
    g = np.array([0.0, 0.0, G0])
    f_ned = a - g
    f_body = np.einsum("tji,tj->ti", Rnb, f_ned)
    # body rates from the DCM derivative: skew(w) = R^T dR/dt
    dR = np.gradient(Rnb, dt, axis=0)
    W = np.einsum("tji,tjk->tik", Rnb, dR)
    w_body = np.stack([W[:, 2, 1], W[:, 0, 2], W[:, 1, 0]], axis=1)
    return Rnb, w_body, f_body, v, p


def integrate(Rnb0, v0, p0, w_body, f_body, dt):
    """Strapdown integration, trapezoidal.

    The derivatives in synthesize() are CENTRAL differences, so a forward-Euler
    integrator does not invert them and the round trip drifts by hundreds of
    metres purely from the scheme mismatch -- which is what GATE 1 caught the
    first time this file was run. Trapezoidal integration is the matching
    inverse to O(dt^2) and closes the loop.
    """
    n = w_body.shape[0]
    R = Rnb0.copy()
    v = v0.copy()
    p = p0.copy()
    P = np.empty((n, 3))
    V = np.empty((n, 3))
    g = np.array([0.0, 0.0, G0])
    a_prev = R @ f_body[0] + g
    for t in range(n):
        P[t] = p
        V[t] = v
        a = R @ f_body[t] + g
        v_new = v + 0.5 * (a_prev + a) * dt
        p = p + 0.5 * (v + v_new) * dt
        v = v_new
        a_prev = a
        # the rate is a CENTRAL difference at t, so the step from t to t+1 wants
        # the rate at t+1/2. Using w[t] alone leaves 0.014 deg of attitude error
        # over ten minutes -- the size of the real tilt uncertainty -- and that
        # tilt turns into 434 m of position through g t^2 / 2. This is the whole
        # reason GATE 1 failed twice.
        w_mid = w_body[t] if t + 1 >= n else 0.5 * (w_body[t] + w_body[t + 1])
        th = w_mid * dt
        ang = np.linalg.norm(th)
        if ang > 1e-12:
            k = th / ang
            K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
            dR = np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * (K @ K)
            R = R @ dR
    return P, V


def main():
    path = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") \
        else "/tmp/line_1007_06_py.h5"
    d = load(path)
    dt = float(d["dt"])
    tlat = np.asarray(d["true_lat"]).ravel()
    tlon = np.asarray(d["true_lon"]).ravel()
    ilat = np.asarray(d["ins_lat"]).ravel()
    ilon = np.asarray(d["ins_lon"]).ravel()
    alt = np.asarray(d["ins_alt"]).ravel()
    n = tlat.size
    lat0, lon0, alt0 = float(tlat[0]), float(tlon[0]), float(alt[0])
    p_true = ned_from_llh(tlat, tlon, alt, lat0, lon0, alt0)
    p_ins = ned_from_llh(ilat, ilon, alt, lat0, lon0, alt0)

    # 10 Hz map-matched truth differentiated twice is dominated by quantization:
    # at a 5 s window the reconstructed specific force ranges over 8.5 to
    # 16.8 m/s^2 where a 16 deg bank implies 9.8 to 10.2, i.e. plus or minus
    # 7 m/s^2 of pure noise. The window has to be long enough to kill that, and
    # the price is that real dynamics shorter than the window are smoothed away
    # -- acceptable on a straight survey line, not on a manoeuvre.
    sw = int(arg("--smooth", 201, int))
    Rnb, w_b, f_b, v_ref, p_ref = synthesize(p_true, dt, sw)

    print(f"{os.path.basename(path)}  {n} samples at {1/dt:g} Hz, "
          f"{n*dt:.0f} s, smoothing window {sw} samples ({sw*dt:.1f} s)")
    print(f"  speed {np.linalg.norm(v_ref[:, :2], axis=1).mean():.1f} m/s, "
          f"bank |max| {np.degrees(np.abs(np.arctan2(np.gradient(np.unwrap(np.arctan2(v_ref[:,1],v_ref[:,0])),dt)*np.hypot(v_ref[:,0],v_ref[:,1]),G0))).max():.1f} deg, "
          f"|w| max {np.abs(w_b).max():.4f} rad/s, "
          f"|f| range {np.linalg.norm(f_b,axis=1).min():.2f}-{np.linalg.norm(f_b,axis=1).max():.2f} m/s^2")

    # ---- GATE 1: round trip on the clean IMU
    P, V = integrate(Rnb[0], v_ref[0], p_ref[0], w_b, f_b, dt)
    err = np.linalg.norm(P[:, :2] - p_ref[:, :2], axis=1)
    print()
    print("GATE 1  clean round trip (integrate the synthetic IMU back)")
    print(f"  horizontal error   final {err[-1]:.3f} m   max {err.max():.3f} m")
    ok1 = err.max() < 1.0
    print("  ->", "PASS" if ok1 else "FAIL -- do not use this IMU for anything")

    # ---- GATE 2: injected errors must reproduce the recorded INS drift
    gb = float(arg("--gyro-bias", 0.24, float)) * DEG_PER_HR
    ab = float(arg("--accel-bias", 12.0, float)) * UG
    rng = np.random.default_rng(7)
    dirg = rng.standard_normal(3); dirg /= np.linalg.norm(dirg)
    dira = rng.standard_normal(3); dira /= np.linalg.norm(dira)
    w_e = w_b + gb * dirg
    f_e = f_b + ab * dira
    v_e = v_ref[0] + 0.1005 * (rng.standard_normal(3) / np.sqrt(3))
    Pe, _ = integrate(Rnb[0], v_e, p_ref[0], w_e, f_e, dt)
    de = np.linalg.norm(Pe[:, :2] - p_true[:, :2], axis=1)
    dr = np.linalg.norm(p_ins[:, :2] - p_true[:, :2], axis=1)
    print()
    print("GATE 2  injected-error drift vs the RECORDED INS drift")
    print(f"  synthetic INS   final {de[-1]:8.1f} m   rms {np.sqrt((de**2).mean()):8.1f} m")
    print(f"  recorded  INS   final {dr[-1]:8.1f} m   rms {np.sqrt((dr**2).mean()):8.1f} m")
    print(f"  ratio           final {de[-1]/max(dr[-1],1e-9):8.2f}       "
          f"rms {np.sqrt((de**2).mean())/np.sqrt((dr**2).mean()):8.2f}")
    ok2 = 0.3 < np.sqrt((de**2).mean()) / max(np.sqrt((dr**2).mean()), 1e-9) < 3.0
    print("  ->", "PASS (same order as the recording)" if ok2 else
          "FAIL -- retune the injected errors before drawing conclusions")
    print()
    print("gyro bias %.3f deg/hr, accel bias %.1f ug, initial velocity error %.3f m/s"
          % (gb / DEG_PER_HR, ab / UG, np.linalg.norm(v_e - v_ref[0])))
    return 0 if (ok1 and ok2) else 1


if __name__ == "__main__":
    sys.exit(main())
