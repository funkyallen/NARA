"""Command-line entry point for raw repeat-pass wireline GR registration."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import numpy as np

from .solver import align_raw_curves


def _load_two_col_csv(path: Path, depth_col: int = 0, val_col: int = 1):
    depths, vals = [], []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or len(row) <= max(depth_col, val_col):
                continue
            try:
                depths.append(float(row[depth_col]))
                vals.append(float(row[val_col]))
            except ValueError:
                continue
    return np.asarray(depths, dtype=float), np.asarray(vals, dtype=float)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Align two repeat-pass wireline GR curves with NARA.")
    ap.add_argument("--ref", required=True, help="Reference pass CSV path (columns: depth_m, gr)")
    ap.add_argument("--tar", required=True, help="Target repeat pass CSV path (columns: depth_m, gr)")
    ap.add_argument("--out", required=True, help="Output aligned CSV path")
    ap.add_argument("--band-m", type=float, default=3.048, help="Max allowed shift search band in metres (default: 3.048)")
    ap.add_argument("--no-gate", action="store_true", help="Disable two-way reciprocity check")
    args = ap.parse_args(argv)

    rd, rv = _load_two_col_csv(Path(args.ref))
    td, tv = _load_two_col_csv(Path(args.tar))
    res = align_raw_curves(rd, rv, td, tv, band_m=args.band_m, use_gate=not args.no_gate, profile=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["depth_m", "ref_gr", "aligned_tar_depth_m", "shift_m", "aligned_tar_gr"])
        for z, rg, td_al, sh, tg in zip(
            res["depth_m"], res["ref_gr"], res["aligned_tar_depth_m"], res["shift_m"], res["aligned_tar_gr"]
        ):
            w.writerow([f"{z:.4f}", f"{rg:.6g}", f"{td_al:.4f}", f"{sh:+.4f}", f"{tg:.6g}"])

    meta = {
        "n_core": int(len(res["depth_m"])),
        "step_m": res["step_m"],
        "lam_eff": round(res["lam_eff"], 4),
        "j_data_pilot": round(res["j_data_pilot"], 5),
        "model_sigma_a_b": res["model"],
        "rev_model_sigma_a_b": res["rev_model"],
        "gate_triggered": res["gate_triggered"],
        "mid_width_005_m": res["mid_width_005_m"],
        "mid_width_norm_m": res["mid_width_norm_m"],
    }
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
