#!/usr/bin/env python3
"""Synthetic MagNav line in the export schema of export_full_line.py.

Purpose is CODE VALIDATION ONLY. The SGL 2020 flight data is a lazy artifact
behind a host some environments cannot reach, so there has to be a way to
exercise run_gtsam_decimated.py / run_gtsam_split.py end to end -- the binary
measurement factors, the compensation-node chaining, the timestamp re-stamping,
and marginalization in the presence of persisted nodes -- without it.

Do NOT read performance conclusions off this file. The measurement is generated
from the same model the estimators assume, so it says nothing about which
parameterization survives real cold-start data; that question is settled by
split_beta.sh on the real lines. What it does establish is that the code runs
and that the control mode reproduces the reference implementation.

Scales are matched to the real lines so the operating point is not absurd:
map |grad| 121 nT/km (real median 125), interference 355 nT rms (real 197 on
Mag 5, 1771 on Mag 4), INS drift 176 m DRMS (real 121-318).

Usage: python make_synth_line.py [out.h5]
"""
import sys
import numpy as np, h5py
from scipy.linalg import expm
rng = np.random.default_rng(7)
RE = 6378137.0
N, dt, nTL = 18000, 0.1, 19
nx = 18 + nTL                       # 17 Pinson + 19 TL + 1 FOGM
lat0, lon0 = np.deg2rad(45.0), np.deg2rad(-76.0)
tau_f = 180.0

# --- truth trajectory: 50 m/s, gentle heading change -------------------------
t = np.arange(N)*dt
hdg = 0.6*np.sin(2*np.pi*t/1200.0)
v = 50.0
true_lat = lat0 + np.cumsum(v*np.cos(hdg)*dt)/RE
true_lon = lon0 + np.cumsum(v*np.sin(hdg)*dt)/(RE*np.cos(lat0))

# --- Pinson-shaped F: position error driven by velocity error ----------------
F = np.zeros((nx, nx))
F[0, 3] = 1.0/RE
F[1, 4] = 1.0/(RE*np.cos(lat0))
F[3, 6] = -9.81; F[4, 7] = 9.81          # vel err from attitude err
F[6, 11] = 1e-3; F[7, 12] = 1e-3         # attitude err from gyro bias
F[nx-1, nx-1] = -1.0/tau_f               # FOGM
Phi1 = expm(F*dt)
Phi = np.repeat(Phi1[None, :, :], N-1, axis=0)

# --- true error state: initial vel/att error -> growing position error -------
x = np.zeros(nx)
x[3] = 0.12; x[4] = -0.09                # m/s velocity error
x[6] = 6e-6; x[7] = -4e-6              # rad attitude error
beta_true = rng.normal(size=nTL)*95.0     # cabin-magnetometer scale
xs = np.zeros((N, nx))
sig_S = 3.0
for k in range(N):
    xs[k] = x
    x = Phi1 @ x
    x[nx-1] += rng.normal()*sig_S*np.sqrt(2*dt/tau_f)
    x[17:36] = beta_true*(1 + 0.15*np.sin(2*np.pi*k*dt/2400.0))   # slow drift
ins_lat = true_lat - xs[:, 0]
ins_lon = true_lon - xs[:, 1]

# --- smooth anomaly map, gradient of order 100-300 nT/km ---------------------
pad = 0.02
glat = np.linspace(true_lat.min()-pad, true_lat.max()+pad, 400)
glon = np.linspace(true_lon.min()-pad, true_lon.max()+pad, 400)
LA, LO = np.meshgrid(glat, glon, indexing="ij")
gh = np.zeros_like(LA)
for (a, kx, ky, ph) in [(520, 900, 700, .3), (330, 1900, 1500, 1.1),
                        (210, 4200, 3300, 2.4), (140, 260, 340, 0.7)]:
    gh += a*np.sin(kx*(LA-glat[0]) + ph)*np.cos(ky*(LO-glon[0]) - ph)

def bilin(lat, lon):
    i = np.clip((lat-glat[0])/(glat[1]-glat[0]), 0, glat.size-1.001)
    j = np.clip((lon-glon[0])/(glon[1]-glon[0]), 0, glon.size-1.001)
    i0 = i.astype(int); j0 = j.astype(int); fi = i-i0; fj = j-j0
    return (gh[i0,j0]*(1-fi)*(1-fj)+gh[i0+1,j0]*fi*(1-fj)
            +gh[i0,j0+1]*(1-fi)*fj+gh[i0+1,j0+1]*fi*fj)

# --- TL regressor and measurement -------------------------------------------
# TL regressors: smooth broadband processes (filtered white noise) rather than
# pure tones, so the 19 columns are not mutually near-degenerate the way a
# sinusoid basis is. Real TL columns come from fluxgate direction cosines.
def smooth(n, lc):
    w = rng.normal(size=n)
    k = np.exp(-0.5*(np.arange(-4*lc, 4*lc+1)/lc)**2); k /= k.sum()
    return np.convolve(w, k, mode="same")
cols = [np.ones(N)]
for i in range(nTL-1):
    lc = 10**rng.uniform(1.7, 3.2)               # 50 s to 100 s correlation
    c = smooth(N, lc); cols.append(c/ (c.std()+1e-12))
A = np.column_stack(cols)
print("A cond =", np.linalg.cond(A.T@A/N))
Rm = 12.0**2
meas = bilin(true_lat, true_lon) + (A*xs[:, 17:36]).sum(1) + xs[:, nx-1] \
       + rng.normal(size=N)*np.sqrt(Rm)

# --- P0 / Qd, block diagonal exactly as create_P0 / create_Qd ----------------
P0 = np.zeros((nx, nx))
P0[0,0] = (0.1/RE)**2; P0[1,1] = (0.1/(RE*np.cos(lat0)))**2; P0[2,2] = 1.0
for i in (3,4,5): P0[i,i] = 1.0**2
for i in range(6,17): P0[i,i] = 1e-6
P0[17:36,17:36] = np.eye(nTL)
P0[nx-1,nx-1] = sig_S**2
Qd = np.zeros((nx, nx))
for i in range(17): Qd[i,i] = 1e-14
Qd[0,0] = Qd[1,1] = 1e-31
Qd[17:36,17:36] = np.eye(nTL)*(1.0*dt)**2
Qd[nx-1,nx-1] = sig_S**2*2*dt/tau_f

with h5py.File((sys.argv[1] if len(sys.argv) > 1 else "/tmp/line_synth_full.h5"), "w") as f:
    for k, val in dict(N=N, nx=nx, nTL=nTL, dt=dt, R=Rm, warm=600.0, win=300.0,
                       ref_drms=np.nan).items():
        f[k] = val
    f["Phi"] = Phi; f["A"] = A; f["meas"] = meas; f["meas_mag5"] = meas
    f["ins_lat"] = ins_lat; f["ins_lon"] = ins_lon
    f["true_lat"] = true_lat; f["true_lon"] = true_lon
    f["P0"] = P0; f["Qd"] = Qd
    f["glat"] = glat; f["glon"] = glon; f["gh_rowmajor"] = gh

dn = (ins_lat-true_lat)*RE; de = (ins_lon-true_lon)*RE*np.cos(true_lat)
m = t >= 600
print(f"N={N} nx={nx} INS drift final {np.hypot(dn,de)[-1]:.0f} m, "
      f"DRMS(warm600) {np.sqrt(np.mean(dn[m]**2+de[m]**2)):.0f} m")
gy,gx = np.gradient(gh,(glat[1]-glat[0])*RE,(glon[1]-glon[0])*RE*np.cos(lat0))
print(f"map |grad| along track median {np.median(np.hypot(gx,gy))*1000:.0f} nT/km")
print(f"interference A.beta rms {np.sqrt(np.mean(((A*xs[:,17:36]).sum(1))**2)):.0f} nT")
