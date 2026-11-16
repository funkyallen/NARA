"""Reproduce Experiment E4 (Tables 6 & 7) from saved archives or full bundled data recomputation."""
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

from nara import MODELS, Graph, correlation, compute_base_costs, dp_with_precomputed, pilot_data_residual, is_same_side_bound_sat, solve_core
from nara.solver import constant_runs_mask, good, sample_aux


def _eval_ablation_field_well(wrec: dict):
    d = np.load(REPO / "data/field_iodp_13wells" / wrec["npz"])
    xf, y = d["x_full"], d["y"]
    rev_xf = d["rev_x_full"] if len(d["rev_x_full"]) else None
    rev_y = d["rev_y"] if len(d["rev_y"]) else None
    origin = float(d["origin_m"][0])
    const_gr = d["const_gr"]

    n = len(xf) - 80
    u = np.arange(-30, n + 30.0)
    z_idx = np.arange(n, dtype=float)
    mp_xc = z_idx + float(correlation(xf[40 : 40 + n], y, u, 20.0))

    g = Graph(n, 0.25, 20, 20.0, 0.0, 40, 30)
    tpl_all, bc_hub1 = compute_base_costs(g, xf, y, MODELS, delta=1.0)
    idx_r0 = np.flatnonzero(MODELS[:, 0] == 0)
    tpl_r0, bc_r0 = tpl_all[idx_r0], [bc[idx_r0] for bc in bc_hub1]

    s_005 = dp_with_precomputed(g, xf, y, MODELS, tpl_all, bc_hub1, 0.05)
    s_200 = dp_with_precomputed(g, xf, y, MODELS, tpl_all, bc_hub1, 2.0)
    jd_ra = pilot_data_residual(g, bc_hub1, s_005)
    s_ung = dp_with_precomputed(g, xf, y, MODELS, tpl_all, bc_hub1, 0.05 + 15.0 * jd_ra)

    s_r0_1 = dp_with_precomputed(g, xf, y, MODELS[idx_r0], tpl_r0, bc_r0, 0.05)
    jd_r0 = pilot_data_residual(g, bc_r0, s_r0_1)
    s_r0 = dp_with_precomputed(g, xf, y, MODELS[idx_r0], tpl_r0, bc_r0, 0.05 + 15.0 * jd_r0)

    _, bc_l2 = compute_base_costs(g, xf, y, MODELS, delta=float("inf"), templates=tpl_all)
    s_base = dp_with_precomputed(g, xf, y, MODELS, tpl_all, bc_l2, 0.01)
    s_l2_1 = dp_with_precomputed(g, xf, y, MODELS, tpl_all, bc_l2, 0.05)
    jd_l2 = pilot_data_residual(g, bc_l2, s_l2_1)
    s_l2 = dp_with_precomputed(g, xf, y, MODELS, tpl_all, bc_l2, 0.05 + 15.0 * jd_l2)

    nrg = False
    if rev_xf is not None and rev_y is not None:
        _, mdl_rev, *_ = solve_core(rev_xf, rev_y, band=20.0)
        nrg = is_same_side_bound_sat(s_ung["model"].tolist(), mdl_rev)

    variants = {
        "XCORR": mp_xc,
        "ABL_Base-RA": s_base["mapped"],
        "ABL_RA-Fix0.05-Hub": s_005["mapped"],
        "ABL_RA-Fix2.0-Hub": s_200["mapped"],
        "ABL_RA-SNR-L2": s_l2["mapped"],
        "ABL_R0-SNR-Hub": s_r0["mapped"],
        "ABL_RA-SNR-Hub": s_ung["mapped"],
        "ABL_RA-SNR-Hub+Gate": mp_xc if nrg else s_ung["mapped"],
    }
    vnames = list(variants.keys())
    wsf_m = origin + np.array([variants[k] for k in vnames]) * 0.1524

    eps = []
    for ci_idx, ch in enumerate(wrec["channels"]):
        ref_v = d[f"ch{ci_idx}_ref_v"]
        tar_d = d["tar_depth"] if "tar_depth" in d else d[f"ch{ci_idx}_tar_depth"]
        tar_v = d[f"ch{ci_idx}_tar_val"]
        pos_only = bool(ch["pos_only"])
        aligned = np.array([sample_aux(tar_d, tar_v, m, float(ch["step_m"]), pos_only) for m in wsf_m])
        common = good(ref_v) & np.all(np.isfinite(aligned), axis=0)
        if pos_only:
            common &= (ref_v > 0) & np.all(aligned > 0, axis=0)

        hrule = ch["healthy_rule"]
        if hrule == "lcal_gt_6":
            lcal_all = np.array([sample_aux(tar_d, d["tar_lcal"], m, float(ch["step_m"]), False) for m in wsf_m])
            screened = common & (d["ref_lcal"] > 6.0) & np.all(lcal_all > 6.0, axis=0) & (~const_gr)
        elif hrule == "chrp_gt_0":
            tc_all = np.array([sample_aux(tar_d, d["tar_chrp"], m, float(ch["step_m"]), False) for m in wsf_m])
            screened = common & (d["ref_chrp"] > 0.0) & np.all(tc_all > 0.0, axis=0)
        else:
            screened = common & (~const_gr) & (~constant_runs_mask(ref_v, 1e-7, 5))

        ratios_all, ratios_scr = {}, {}
        r_xc_all = float(np.sqrt(np.mean((aligned[0, common] - ref_v[common]) ** 2)))
        r_xc_scr = float(np.sqrt(np.mean((aligned[0, screened] - ref_v[screened]) ** 2)))
        for idx, k in enumerate(vnames[1:], start=1):
            ratios_all[k] = float(np.sqrt(np.mean((aligned[idx, common] - ref_v[common]) ** 2)) / r_xc_all)
            ratios_scr[k] = float(np.sqrt(np.mean((aligned[idx, screened] - ref_v[screened]) ** 2)) / r_xc_scr)
        eps.append({"cohort": wrec["cohort"], "case_id": ch["case_id"], "all": ratios_all, "scr": ratios_scr})
    return eps


def recompute_ablations(ref_summary: dict, workers: int):
    t0 = time.perf_counter()
    manifest = json.loads((REPO / "data/field_iodp_13wells/cohort_manifest.json").read_text(encoding="utf-8"))
    all_ep = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for res in pool.map(_eval_ablation_field_well, manifest):
            all_ep.extend(res)
    sub_b = [e for e in all_ep if e["cohort"] == "B_eval"]
    max_diff = 0.0
    print(f"\n=== Live Recomputed Table 6 Field Ablations ({time.perf_counter() - t0:.1f} s) ===")
    for k in ("ABL_Base-RA", "ABL_RA-Fix0.05-Hub", "ABL_RA-Fix2.0-Hub", "ABL_RA-SNR-L2", "ABL_R0-SNR-Hub", "ABL_RA-SNR-Hub", "ABL_RA-SNR-Hub+Gate"):
        ra = np.array([e["all"][k] for e in all_ep])
        rb = np.array([e["scr"][k] for e in sub_b])
        wins = int(np.sum(ra < 1.0 - 1e-12))
        gt5 = int(np.sum(ra > 1.05))
        bwins = int(np.sum(rb < 1.0 - 1e-12))
        max_diff = max(max_diff, abs(float(ra.mean()) - ref_summary[k]["all19_mean_ratio"]), abs(float(rb.mean()) - ref_summary[k]["blind15_hth_mean_ratio"]))
        print(f"{k:<22} | All19={wins:2d}/19 (gt5%={gt5:2d}, mean={ra.mean():.4f}) | Blind-Scr={bwins:2d}/15 (mean={rb.mean():.4f})")
    assert max_diff <= 1e-3, f"E4 field recompute parity failed: {max_diff:.3e}"
    print(f"Recompute parity PASSED (max mean ratio diff = {max_diff:.3e})")


def print_archive_tables(s: dict):
    print("=== Table 6: NARA Framework Component Ablations ===")
    rows = [
        ("Base-RA (0.5*e^2, lam=0.01)", "ABL_Base-RA"),
        ("w/o Adapt (Fixed lam=0.05)", "ABL_RA-Fix0.05-Hub"),
        ("w/o Adapt (Fixed lam=2.00)", "ABL_RA-Fix2.0-Hub"),
        ("w/o Huber (L2 Loss)", "ABL_RA-SNR-L2"),
        ("w/o Broadening (9-bank, sig=0)", "ABL_R0-SNR-Hub"),
        ("w/o Gate (Un-gated DP)", "ABL_RA-SNR-Hub"),
        ("NARA (Full Framework)", "ABL_RA-SNR-Hub+Gate"),
    ]
    for name, key in rows:
        r = s[key]
        print(
            f"{name:<30} | Prim={r['primary216_mae']:.4f} | Box={r['box54_mae']:.4f} | "
            f"All19={r['all19_wins']}/19 (gt5%={r['all19_gt5']}, mean={r['all19_mean_ratio']:.4f}) | "
            f"Blind-Scr={r['blind15_hth_wins']}/15 ({r['blind15_hth_mean_ratio']:.4f})"
        )

    print("\n=== Table 7: Four-Axis Parameter Sensitivity (Un-gated Core DP) ===")
    axes = [
        ("delta", [("0.5", "SEN_DELTA_0.5"), ("1.0*", "SEN_DELTA_1.0"), ("1.5", "SEN_DELTA_1.5"), ("2.0", "SEN_DELTA_2.0"), ("inf", "SEN_DELTA_inf")]),
        ("kappa", [("0", "SEN_KAPPA_0"), ("5", "SEN_KAPPA_5"), ("10", "SEN_KAPPA_10"), ("15*", "SEN_KAPPA_15"), ("20", "SEN_KAPPA_20"), ("30", "SEN_KAPPA_30")]),
        ("lam0", [("0.01", "SEN_LAM0_0.01"), ("0.05*", "SEN_LAM0_0.05"), ("0.10", "SEN_LAM0_0.10"), ("0.20", "SEN_LAM0_0.20")]),
        ("(h,dg)", [("(10,0.25)", "SEN_DISC_10_0.25"), ("(20,0.25)*", "SEN_DISC_20_0.25"), ("(40,0.25)", "SEN_DISC_40_0.25"), ("(20,0.50)", "SEN_DISC_20_0.50")]),
    ]
    for axis, items in axes:
        for lbl, key in items:
            r = s[key]
            print(
                f"{axis:<6} {lbl:<11} | Prim={r['primary216_mae']:.4f} | Box={r['box54_mae']:.4f} | "
                f"All19={r['all19_wins']}/19 (gt5%={r['all19_gt5']}, mean={r['all19_mean_ratio']:.4f}) | Blind-Scr={r['blind15_hth_mean_ratio']:.4f}"
            )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recompute-ablations", action="store_true", help="Recompute all Table 6 field ablations live from bundled data/")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    e4 = json.loads((REPO / "results/reference_e4_ablation_sensitivity.json").read_text(encoding="utf-8"))
    print_archive_tables(e4["summary"])
    if args.recompute_ablations:
        recompute_ablations(e4["summary"], args.workers)


if __name__ == "__main__":
    main()
