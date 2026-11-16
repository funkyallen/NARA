"""Quickstart demo: run NARA on two IODP holes (U1611A adaptive alignment & U1437D reciprocity gate)."""
import sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from nara import align_nara, correlation
from nara.solver import constant_runs_mask, sample_aux


def demo_well(well: str):
    d = np.load(REPO / "data/field_iodp_13wells" / f"{well}.npz")
    xf, y = d["x_full"], d["y"]
    rev_xf, rev_y = d["rev_x_full"], d["rev_y"]
    origin = float(d["origin_m"][0])
    n = len(xf) - 80

    # 1) Baseline rigid shift (XCORR)
    d_xc = float(correlation(xf[40 : 40 + n], y, np.arange(-30, n + 30.0), 20.0))
    mp_xc = np.arange(n, dtype=float) + d_xc

    # 2) NARA (with two-way reciprocity check & whole-map conditional profile Q_k(v))
    res = align_nara(
        xf, y, band=20.0,
        rev_xf=rev_xf if len(rev_xf) else None,
        rev_y=rev_y if len(rev_y) else None,
        profile=True,
    )

    # 3) Score held-out RLA5 channel
    ref_v = d["ch0_ref_v"]
    tar_d, tar_x = d["ch0_tar_depth"], d["ch0_tar_val"]
    al_xc = sample_aux(tar_d, tar_x, origin + mp_xc * 0.1524, 0.0508, True)
    al_na = sample_aux(tar_d, tar_x, origin + res["mapped"] * 0.1524, 0.0508, True)
    valid = np.isfinite(ref_v) & np.isfinite(al_xc) & np.isfinite(al_na) & (ref_v > 0) & (al_xc > 0) & (al_na > 0)
    rmse_xc = float(np.sqrt(np.mean((al_xc[valid] - ref_v[valid]) ** 2)))
    rmse_na = float(np.sqrt(np.mean((al_na[valid] - ref_v[valid]) ** 2)))

    # Conditional 0.005-sublevel width at midpoint knot
    g = res["graph"]
    q = res["solution"]["q"]
    k_mid = g.K // 2
    dq = q[k_mid].min(axis=0) - res["solution"]["value"]
    sub = g.grid[dq <= 0.005] * 0.1524
    width_m = float(sub[-1] - sub[0]) if len(sub) else 0.0

    print(f"=== IODP Hole {well} (n={n}) ===")
    print(f"  Pilot J_data     : {res['j_data_pilot']:.4f}  ->  lam_eff = {res['lam_eff']:.2f}")
    print(f"  Fwd / Rev model  : {res['model']} / {res['rev_model']}")
    print(f"  Reciprocity gate : {'TRIGGERED (reverted to XCORR)' if res['gate_triggered'] else 'Passed (using NARA DP map)'}")
    print(f"  Midpoint Q width : {width_m:.2f} m (Delta Q <= 0.005)")
    print(f"  Held-out RLA5    : XCORR RMSE = {rmse_xc:.4f}  |  NARA RMSE = {rmse_na:.4f}\n")


if __name__ == "__main__":
    demo_well("U1611A")
    demo_well("U1437D")
