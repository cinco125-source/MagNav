#!/usr/bin/env python3
"""Export raw SGL 2020 channels for a line, a calibration box, or a time range.

Standalone by design. research/gtsam_poc/export_full_line.py builds a factor
graph's ingredients and is bound to them -- it needs a map file, borrows P0/Qd
from a committed segment, computes a Pinson tensor, and imports ppigrf and
scipy. None of that belongs in another paper's toolchain. This script reads the
flight HDF5 and writes channels, nothing else: numpy, h5py, and the standard
library.

WHAT IT WRITES
  scalar magnetometers  mag_1_c mag_1_uc mag_2_c/uc mag_3_c/uc mag_4_uc
                        mag_5_uc mag_6_uc
  fluxgates             flux_a/b/c/d, each x y z t (raw, not a fitted basis)
  currents              the 14 cur_* channels
  voltages              the vol_* rails, including vol_cabt (cabin temperature)
  auxiliary             diurnal, igrf, ogs_mag, ogs_alt, and whatever else of
                        COMP_AUX the flight records
  trajectory            tt, lat, lon, alt (truth), and the INS series
  attitude              ins_roll ins_pitch ins_yaw [rad]

Channels a flight does not record are skipped and named. mag_1_c and flux_a get
an explicit warning when absent, since those two decide what can be claimed
rather than merely what is available: without the stinger there is no clean
reference to score a compensated cabin sensor against.

WINDOWS come from examples/dataframes, so any line in the survey is reachable
rather than the five that were hard-coded for the navigation study:
  df_all.csv   every line, t_start/t_end        -> --line
  df_cal.csv   calibration boxes                -> --cal
  df_nav.csv   navigation lines, map and line_alt (reported by --list)

Altitude pairs are in df_nav: Flt1003 flies 1003.02 at 398 m and 1003.04 at
800 m over the same Eastern_395 map, and Flt1007 flies 1007.01/1007.02 at
798/799 m. An altitude-transfer study is a matter of exporting both, not of
missing data.

USAGE
  # what is in a flight
  export_sgl_channels.py Flt1003_train.h5 --flight Flt1003 --list

  # one survey line, HDF5 and CSV
  export_sgl_channels.py Flt1003_train.h5 --flight Flt1003 --line 1003.02 \\
      --out comp_1003_02.h5 --csv comp_1003_02.csv

  # the altitude pair
  export_sgl_channels.py Flt1003_train.h5 --flight Flt1003 \\
      --line 1003.02 --line 1003.04 --out comp_1003_alt.h5

  # every calibration box in a flight, one group per box
  export_sgl_channels.py Flt1006_train.h5 --flight Flt1006 --cal \\
      --out cal_1006.h5

Multiple windows are written as HDF5 groups named by line and index; a single
window is written flat. CSV always writes one file per window, suffixed when
there is more than one.
"""
import csv
import os
import sys

import numpy as np
import h5py

HERE = os.path.dirname(os.path.abspath(__file__))
DF_DIR = os.path.join(HERE, "..", "examples", "dataframes")

MAGS = ["mag_1_c", "mag_1_uc", "mag_2_c", "mag_2_uc", "mag_3_c", "mag_3_uc",
        "mag_4_uc", "mag_5_uc", "mag_6_uc"]
FLUX = [f"flux_{b}_{c}" for b in "abcd" for c in "xyzt"]
CUR = ["cur_com_1", "cur_ac_hi", "cur_ac_lo", "cur_tank", "cur_flap", "cur_strb",
       "cur_srvo_o", "cur_srvo_m", "cur_srvo_i", "cur_heat", "cur_acpwr",
       "cur_outpwr", "cur_bat_1", "cur_bat_2"]
VOL = ["vol_acc_p", "vol_acc_n", "vol_block", "vol_back", "vol_back_p",
       "vol_back_n", "vol_srvo", "vol_cabt", "vol_fan", "vol_bat_1", "vol_bat_2",
       "vol_res_p", "vol_res_n", "vol_outpwr", "vol_acpwr", "vol_gyro_1",
       "vol_gyro_2"]
AUX = ["diurnal", "igrf", "ogs_mag", "ogs_alt", "baro", "radar", "topo", "dem",
       "drape", "utm_x", "utm_y", "utm_z", "msl", "lat", "lon", "alt", "line"]
TRAJ = ["tt", "ins_lat", "ins_lon", "ins_alt", "ins_vn", "ins_vw", "ins_vu",
        "ins_roll", "ins_pitch", "ins_yaw", "ins_wander",
        "ins_acc_x", "ins_acc_y", "ins_acc_z"]
ALL_FIELDS = MAGS + FLUX + CUR + VOL + AUX + TRAJ
CRITICAL = {"mag_1_c": "no clean stinger reference to score a compensated sensor against",
            "flux_a_x": "no flux A boom, so any basis built on it is unavailable"}


def read_df(name):
    path = os.path.join(DF_DIR, name)
    with open(path, newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def windows(flight, lines, use_cal, tt_range):
    """Resolve the requested windows to (label, t_start, t_end) triples."""
    if tt_range:
        return [("tt", float(tt_range[0]), float(tt_range[1]))]
    if use_cal:
        rows = [r for r in read_df("df_cal.csv") if r["flight"] == flight]
        if not rows:
            sys.exit(f"no calibration boxes for {flight} in df_cal.csv")
        return [(f"cal_{r['line'].replace('.', '_')}_{i}",
                 float(r["t_start"]), float(r["t_end"]))
                for i, r in enumerate(rows)]
    rows = [r for r in read_df("df_all.csv") if r["flight"] == flight]
    by_line = {r["line"]: r for r in rows}
    out = []
    for ln in lines:
        key = ln if ln in by_line else f"{float(ln):.2f}"
        if key not in by_line:
            sys.exit(f"line {ln} not in df_all.csv for {flight}; "
                     f"available: {' '.join(sorted(by_line))}")
        r = by_line[key]
        out.append((f"line_{key.replace('.', '_')}",
                    float(r["t_start"]), float(r["t_end"])))
    return out


def do_list(flight):
    nav = {(r["flight"], r["line"]): r for r in read_df("df_nav.csv")}
    print(f"{flight}: lines in df_all.csv")
    print(f"  {'line':10s} {'t_start':>10s} {'t_end':>10s} {'min':>6s}  "
          f"{'map':14s} {'alt[m]':>6s}")
    for r in read_df("df_all.csv"):
        if r["flight"] != flight:
            continue
        t0, t1 = float(r["t_start"]), float(r["t_end"])
        n = nav.get((flight, r["line"]))
        print(f"  {r['line']:10s} {t0:10.1f} {t1:10.1f} {(t1-t0)/60:6.1f}  "
              f"{(n['map_name'] if n else '-'):14s} {(n['line_alt'] if n else '-'):>6s}")
    cal = [r for r in read_df("df_cal.csv") if r["flight"] == flight]
    print(f"\n{flight}: {len(cal)} calibration box(es) in df_cal.csv")
    for r in cal:
        t0, t1 = float(r["t_start"]), float(r["t_end"])
        print(f"  {r['line']:10s} {t0:10.1f} {t1:10.1f} {(t1-t0)/60:6.1f} min "
              f"{r['map_name'] or '-'}")


def extract(f, t0, t1, line_no=None):
    tt = np.asarray(f["tt"][()]).ravel()
    ind = (tt >= t0) & (tt <= t1)
    if line_no is not None and "line" in f:
        ind &= np.round(np.asarray(f["line"][()]).ravel(), 2) == line_no
    out, missing = {}, []
    for k in ALL_FIELDS:
        if k in f:
            v = np.asarray(f[k][()]).ravel()
            if v.size == tt.size:
                out[k] = v[ind].astype(float)
            else:                                  # scalar attribute, keep as is
                out[k] = v.astype(float)
        else:
            missing.append(k)
    return out, missing, int(ind.sum())


def main():
    a = sys.argv
    if len(a) < 2 or a[1].startswith("--"):
        sys.exit(__doc__)
    flt_path = a[1]
    flight = a[a.index("--flight") + 1] if "--flight" in a else None
    if flight is None:
        flight = os.path.basename(flt_path).split("_")[0]
    if "--list" in a:
        return do_list(flight)
    lines = [a[i + 1] for i, v in enumerate(a) if v == "--line"]
    use_cal = "--cal" in a
    tt_range = (a[a.index("--tt") + 1], a[a.index("--tt") + 2]) if "--tt" in a else None
    if not (lines or use_cal or tt_range):
        sys.exit("give --line LINE (repeatable), --cal, or --tt T0 T1 "
                 "(or --list to see what is available)")
    out_h5 = a[a.index("--out") + 1] if "--out" in a else None
    out_csv = a[a.index("--csv") + 1] if "--csv" in a else None
    if not (out_h5 or out_csv):
        sys.exit("give --out FILE.h5 and/or --csv FILE.csv")

    wins = windows(flight, lines, use_cal, tt_range)
    print(f"{flight}: {len(wins)} window(s)", flush=True)

    results = []
    with h5py.File(flt_path, "r") as f:
        for label, t0, t1 in wins:
            line_no = None
            if label.startswith("line_"):
                line_no = round(float(label[5:].replace("_", ".")), 2)
            d, missing, n = extract(f, t0, t1, line_no)
            if n == 0:
                print(f"  {label}: EMPTY window, skipped", flush=True)
                continue
            dt = float(f["dt"][()]) if "dt" in f else float("nan")
            print(f"  {label}: N={n} ({n*dt/60:.1f} min), "
                  f"{len(d)} channels, {len(missing)} absent", flush=True)
            for k, why in CRITICAL.items():
                if k in missing:
                    print(f"    WARNING: {k} absent -- {why}", flush=True)
            results.append((label, d, n, dt, t0, t1))

    if not results:
        sys.exit("nothing to write")

    if out_h5:
        with h5py.File(out_h5, "w") as o:
            for label, d, n, dt, t0, t1 in results:
                g = o if len(results) == 1 else o.create_group(label)
                for k, v in d.items():
                    g.create_dataset(k, data=v, compression="gzip",
                                     compression_opts=1)
                g.attrs["N"] = n; g.attrs["dt"] = dt
                g.attrs["t_start"] = t0; g.attrs["t_end"] = t1
                g.attrs["flight"] = flight; g.attrs["label"] = label
        print(f"wrote {out_h5}"
              f"{'' if len(results) == 1 else f' ({len(results)} groups)'}",
              flush=True)

    if out_csv:
        base, ext = os.path.splitext(out_csv)
        for label, d, n, dt, t0, t1 in results:
            path = out_csv if len(results) == 1 else f"{base}_{label}{ext}"
            keys = [k for k in ALL_FIELDS if k in d and d[k].size == n]
            with open(path, "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(keys)
                cols = [d[k] for k in keys]
                for i in range(n):
                    w.writerow([f"{c[i]:.10g}" for c in cols])
            print(f"wrote {path}  ({n} rows x {len(keys)} cols)", flush=True)


if __name__ == "__main__":
    main()
