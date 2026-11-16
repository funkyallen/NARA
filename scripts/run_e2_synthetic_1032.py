"""Evaluate NARA across all 1,032 synthetic cases in data/synthetic_1032/."""
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"
import argparse, json, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from nara import align_nara


def _eval_case(rec: dict):
    base = REPO / "data/synthetic_1032"
    ob = np.load(base / rec["obs"])
    tr = np.load(base / rec["truth"])["mapped"]
    res = align_nara(ob["x_full"], ob["y"], band=16.0, use_gate=False)
    err = np.abs(res["mapped"] - tr)
    return {
        "group": rec["group"],
        "well": rec["well"],
        "sigma": rec.get("sigma", ""),
        "mae": float(err.mean()),
        "sig0": float(res["model"][0] == 0.0),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    manifest = json.loads((REPO / "data/synthetic_1032/manifest.json").read_text(encoding="utf-8"))
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(_eval_case, manifest))
    print(f"Evaluated {len(rows)} synthetic cases in {time.perf_counter() - t0:.1f} s\n")

    order = [
        ("Primary (216)", "Primary"),
        ("Ext Stress (108)", "ExtStress"),
        ("Misspec Box (54)", "Misspec_Box"),
        ("Misspec Tri+AR1 (54)", "Misspec_TriAR1"),
        ("Misspec Vary+Het (54)", "Misspec_VaryHet"),
        ("Misspec Asym (54)", "Misspec_Asym"),
        ("U1580A Symm (48)", "U1580A_Symm"),
        ("U1580A Asym (12)", "U1580A_Asym"),
        ("B432 Identity (108)", "B432_identity"),
        ("B432 Large (108)", "B432_large"),
        ("B432 Mild (108)", "B432_mild"),
        ("B432 Shift 0.35 (108)", "B432_shift035"),
    ]
    for label, gkey in order:
        sub = [r for r in rows if r["group"] == gkey]
        wells = sorted(set(r["well"] for r in sub))
        w_mae = float(np.mean([np.mean([r["mae"] for r in sub if r["well"] == w]) for w in wells]))
        print(f"{label:<24} (n={len(sub):3d}, wells={len(wells)}): NARA well-equal MAE = {w_mae:.4f}")

    sig0_rate = float(np.mean([r["sig0"] for r in rows if r["group"].startswith("B432_") and r["sigma"] == "0.0"]))
    print(f"\nB432 zero-blur (sigma=0) exact activation rate: {sig0_rate * 100:.1f}%")


if __name__ == "__main__":
    main()
