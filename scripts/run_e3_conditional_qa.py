"""Reproduce Experiment E3: Exact Q_k(v) excess-cost parity & non-oracle ALT8 global comparison."""
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"
import argparse, json, sys, time
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from nara import MODELS, align_nara, alternating_huber_dp
from nara.solver import eval_continuous_obj


def heuristic_8_starts(g):
    """Return the exact eight fixed deterministic heuristic starts (no DP oracle start)."""
    k, b = g.K, g.band
    return [
        np.zeros(k),
        np.full(k, -b / 2.0),
        np.full(k, b / 2.0),
        np.linspace(-b / 2.0, b / 2.0, k),
        np.linspace(b / 2.0, -b / 2.0, k),
        np.full(k, -b / 4.0),
        np.full(k, b / 4.0),
        np.sin(np.linspace(0.0, 2.0 * np.pi, k)) * (b / 3.0),
    ]


def check_signal(name: str, xf: np.ndarray, y: np.ndarray, band: float, sublevel_th: float = 0.005, max_states: int = None):
    t0 = time.perf_counter()
    res = align_nara(xf, y, band=band, use_gate=False, profile=True)
    dt = time.perf_counter() - t0
    g, sol = res["graph"], res["solution"]

    max_err = 0.0
    checked = 0
    for k in range(g.K):
        for v in range(g.S):
            m, p, qkv = g.witness(sol, k, v)
            if not np.isfinite(qkv):
                continue
            jval = eval_continuous_obj(g, xf, y, MODELS[m], g.grid[p], res["lam_eff"], delta=1.0)
            max_err = max(max_err, abs(qkv - jval))
            checked += 1
            if max_states and checked >= max_states:
                break
        if max_states and checked >= max_states:
            break

    starts = heuristic_8_starts(g)
    alt = alternating_huber_dp(g, res["templates"], res["base_costs"], starts, res["lam_eff"], xf, y)
    gap = float(alt["value"] - sol["value"])

    k_mid = g.K // 2
    dq = sol["q"][k_mid].min(axis=0) - sol["value"]
    sub = g.grid[dq <= sublevel_th] * 0.1524
    w_m = float(sub[-1] - sub[0]) if len(sub) else 0.0
    d_mid = float(g.grid[sol["path"][k_mid]] * 0.1524)
    return {
        "name": name,
        "K": g.K,
        "states": checked,
        "lam_eff": res["lam_eff"],
        "d_mid_m": d_mid,
        "width_m": w_m,
        "max_parity_err": max_err,
        "alt8_gap": gap,
        "opt_val": float(sol["value"]),
        "cpu_sec": dt,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-synthetic", action="store_true", help="Also verify the 90 synthetic E3 cases (36 Primary + 54 Asym)")
    ap.add_argument("--verify", action="store_true", help="Assert exact parity against results/reference_e3_conditional_qa.json")
    args = ap.parse_args()

    ref = json.loads((REPO / "results/reference_e3_conditional_qa.json").read_text(encoding="utf-8"))
    print("=== Field Boreholes (Deterministic Non-Oracle ALT8 & Exact Q_k Parity) ===")
    field_out = {}
    for well in ("U1580A", "U1473A", "U1445A", "U1437D", "U1611A", "U1513E"):
        d = np.load(REPO / "data/field_iodp_13wells" / f"{well}.npz")
        r = check_signal(well, d["x_full"], d["y"], band=20.0, sublevel_th=0.005)
        field_out[well] = r
        tag = " (gate-rejected DP)" if well == "U1437D" else ""
        print(
            f"{well + tag:<22} | K={r['K']:3d} | lam={r['lam_eff']:5.2f} | "
            f"d_mid={r['d_mid_m']:+5.2f}m w0.005={r['width_m']:4.2f}m | "
            f"max|Q-J|={r['max_parity_err']:.2e} | ALT8 gap={r['alt8_gap']:+.2e} | Fwd CPU={r['cpu_sec']:5.2f}s"
        )

    if args.with_synthetic or args.verify:
        manifest = json.loads((REPO / "data/synthetic_1032/manifest.json").read_text(encoding="utf-8"))
        rec_by_id = {m["id"]: m for m in manifest}
        max_syn_err = 0.0
        n_gt1e4 = 0
        total_states = 0
        for row in ref["synthetic_parity_alt8"]:
            m = rec_by_id[row["id"]]
            ob = np.load(REPO / "data/synthetic_1032" / m["obs"])
            r = check_signal(row["id"], ob["x_full"], ob["y"], band=16.0, sublevel_th=0.01)
            max_syn_err = max(max_syn_err, r["max_parity_err"])
            total_states += r["states"]
            if r["alt8_gap"] > 1e-4:
                n_gt1e4 += 1
        print(
            f"\nSynthetic E3 (n={len(ref['synthetic_parity_alt8'])}): "
            f"checked {total_states} states, max|Q-J|={max_syn_err:.2e}, ALT8 gap>1e-4 in {n_gt1e4}/{len(ref['synthetic_parity_alt8'])} cases"
        )
        assert max_syn_err <= 1e-12, f"Synthetic Q-J parity failed: {max_syn_err:.3e}"

    if args.verify:
        for well, r in field_out.items():
            rr = ref["field_profiles"][well]
            assert r["max_parity_err"] <= 1e-12, f"{well} parity failed: {r['max_parity_err']}"
            assert abs(r["opt_val"] - rr["opt_val"]) <= 1e-6, f"{well} opt_val mismatch"
            assert abs(r["width_m"] - rr["mid_width_005_m"]) <= 1e-2, f"{well} width_m mismatch"
        print("Verification PASSED: exact Q_k(v) parity and field profile match.")


if __name__ == "__main__":
    main()
