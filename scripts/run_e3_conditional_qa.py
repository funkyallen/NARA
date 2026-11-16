"""Reproduce Experiment E3: Exact Q_k(v) whole-interval conditional profiles & ALT8 global agreement."""
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"
import json, sys, time
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from nara import MODELS, align_nara, alternating_huber_dp, correlation
from nara.solver import eval_continuous_obj


def check_hole(well: str):
    d = np.load(REPO / "data/field_iodp_13wells" / f"{well}.npz")
    xf, y = d["x_full"], d["y"]
    t0 = time.perf_counter()
    res = align_nara(xf, y, band=20.0, use_gate=False, profile=True)
    dt = time.perf_counter() - t0
    g, sol = res["graph"], res["solution"]

    max_err = 0.0
    for k in range(g.K):
        for v in range(g.S):
            m, p, qkv = g.witness(sol, k, v)
            if not np.isfinite(qkv):
                continue
            jval = eval_continuous_obj(g, xf, y, MODELS[m], g.grid[p], res["lam_eff"], delta=1.0)
            max_err = max(max_err, abs(qkv - jval))

    dxc = float(correlation(xf[40 : 40 + g.n], y, g.u, 20.0))
    starts = [np.zeros(g.K), np.full(g.K, dxc), g.grid[sol["path"]]]
    # Deterministic 8-start perturbation set + DP start check
    rng = np.random.default_rng(20261116)
    for _ in range(5):
        starts.append(np.clip(dxc + rng.uniform(-2.0, 2.0, size=g.K), -18.0, 18.0))
    alt = alternating_huber_dp(g, res["templates"], res["base_costs"], starts, res["lam_eff"], xf, y)
    gap = float(alt["value"] - sol["value"])

    k_mid = g.K // 2
    dq = sol["q"][k_mid].min(axis=0) - sol["value"]
    sub = g.grid[dq <= 0.005] * 0.1524
    w_m = float(sub[-1] - sub[0]) if len(sub) else 0.0
    d_mid = float(g.grid[sol["path"][k_mid]] * 0.1524)
    print(f"{well:<8} | K={g.K:3d} | lam={res['lam_eff']:5.2f} | d_mid={d_mid:+5.2f}m w0.005={w_m:4.2f}m | max|Q-J|={max_err:.2e} | ALT8 gap={gap:+.2e} | CPU={dt:5.2f}s")


def main():
    for well in ("U1580A", "U1473A", "U1445A", "U1437D", "U1611A", "U1513E"):
        check_hole(well)


if __name__ == "__main__":
    main()
