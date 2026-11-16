"""Evaluate NARA (and optional unclipped/9-bank ablations) across all 1,032 synthetic cases."""
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

from nara import MODELS, Graph, align_nara, solve_core


def _eval_case(item: tuple):
    rec, with_abl = item
    base = REPO / "data/synthetic_1032"
    ob = np.load(base / rec["obs"])
    tr = np.load(base / rec["truth"])["mapped"]
    xf, y = ob["x_full"], ob["y"]
    res = align_nara(xf, y, band=16.0, use_gate=False)
    out = {
        "group": rec["group"],
        "well": rec["well"],
        "sigma": rec.get("sigma", ""),
        "mae": float(np.mean(np.abs(res["mapped"] - tr))),
        "sig0": float(res["model"][0] == 0.0),
    }
    if with_abl:
        g = Graph(len(xf) - 80, 0.25, 20, 16.0, 0.01, 40, 30)
        mp_bra = g.solve(xf, y, MODELS)["mapped"]
        mp_r0, *_ = solve_core(xf, y, band=16.0, r0_only=True, adaptive=True)
        out["mae_base_ra"] = float(np.mean(np.abs(mp_bra - tr)))
        out["mae_r0_snr"] = float(np.mean(np.abs(mp_r0 - tr)))
    return out


def _well_equal(sub: list, field: str = "mae") -> tuple:
    wells = sorted(set(r["well"] for r in sub))
    vals = [float(np.mean([r[field] for r in sub if r["well"] == w])) for w in wells]
    return len(wells), float(np.mean(vals)), vals


def _paired_well_bootstrap(vals_a, vals_b, seed: int = 20261116, n_boot: int = 10000):
    diffs = np.asarray(vals_a) - np.asarray(vals_b)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diffs), size=(n_boot, len(diffs)))
    means = diffs[idx].mean(axis=1)
    return float(diffs.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--with-ablations", action="store_true", help="Also run Base-RA and w/o Broadening (9-bank) for 9-well stats")
    ap.add_argument("--verify", action="store_true", help="Assert exact parity against results/reference_e2_synthetic_1032.json")
    args = ap.parse_args()

    manifest = json.loads((REPO / "data/synthetic_1032/manifest.json").read_text(encoding="utf-8"))
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(_eval_case, [(r, args.with_ablations) for r in manifest]))
    print(f"Evaluated {len(rows)} synthetic cases in {time.perf_counter() - t0:.1f} s\n")

    prim = [r for r in rows if r["group"] == "Primary"]
    dev_wells = {f"aligned_well_0{i}" for i in range(1, 7)}
    prim_dev = [r for r in prim if r["well"] in dev_wells]
    prim_held = [r for r in prim if r["well"] not in dev_wells]

    print("=== Primary Benchmark Dev (wells 01-06) vs Held-out (wells 07-09) ===")
    for lbl, sub in (("Primary Dev 01-06 (144)", prim_dev), ("Primary Held-out 07-09 (72)", prim_held), ("Primary All 01-09 (216)", prim)):
        nw, mae_n, wn = _well_equal(sub, "mae")
        if args.with_ablations:
            _, mae_r0, wr0 = _well_equal(sub, "mae_r0_snr")
            _, mae_bra, _ = _well_equal(sub, "mae_base_ra")
            wins = sum(a < b for a, b in zip(wn, wr0))
            print(f"{lbl:<28} (wells={nw}): NARA={mae_n:.4f} | w/o Broadening={mae_r0:.4f} | Base-RA={mae_bra:.4f} | NARA wins={wins}/{nw}")
        else:
            print(f"{lbl:<28} (wells={nw}): NARA={mae_n:.4f}")

    if args.with_ablations:
        _, _, wn = _well_equal(prim, "mae")
        _, _, wr0 = _well_equal(prim, "mae_r0_snr")
        dm, lo, hi = _paired_well_bootstrap(wn, wr0)
        print(f"Primary 216 9-well paired diff (NARA - w/o Broadening): {dm:+.4f} (95% CI [{lo:+.4f}, {hi:+.4f}])\n")

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
    print("=== All 12 Benchmark Groups ===")
    computed_groups = {}
    for label, gkey in order:
        sub = [r for r in rows if r["group"] == gkey]
        nw, w_mae, _ = _well_equal(sub, "mae")
        computed_groups[label] = w_mae
        computed_groups[gkey] = w_mae
        print(f"{label:<24} (n={len(sub):3d}, wells={nw}): NARA well-equal MAE = {w_mae:.4f}")

    unblurred = [r for r in rows if r["group"].startswith("B432_") and r["sigma"] == "0.0"]
    sig0_rate = float(np.mean([r["sig0"] for r in unblurred]))
    print(f"\nB432 zero-blur (sigma=0) exact activation rate: {sig0_rate * 100:.1f}% (n={len(unblurred)} unblurred cases)")

    if args.verify:
        ref = json.loads((REPO / "results/reference_e2_synthetic_1032.json").read_text(encoding="utf-8"))
        max_diff = 0.0
        for gkey, d in ref["summary_600"].items():
            max_diff = max(max_diff, abs(computed_groups[gkey] - d["RA_SNR_HUB"]["well_equal_mae"]))
        for gkey, d in ref["summary_b432"].items():
            max_diff = max(max_diff, abs(computed_groups[f"B432_{gkey}"] - d["RA_SNR_HUB"]["well_equal_mae"]))
        assert max_diff <= 1e-6, f"Synthetic MAE parity failed: {max_diff:.3e}"
        assert len(unblurred) == 216 and abs(sig0_rate - 0.5787037) <= 1e-4, f"Sigma=0 rate mismatch: {sig0_rate}"
        print(f"Verification PASSED: max well-equal MAE diff = {max_diff:.3e}")


if __name__ == "__main__":
    main()
