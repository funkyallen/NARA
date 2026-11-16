"""Evaluate NARA and baselines across the 13 IODP boreholes (19 held-out endpoints)."""
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

from nara import MODELS, Graph, align_nara, correlation, slope_dtw, solve_core
from nara.graph import rcorr_length_map
from nara.solver import constant_runs_mask, good, sample_aux, score_metrics


def eval_well(wrec: dict):
    d = np.load(REPO / "data/field_iodp_13wells" / wrec["npz"])
    xf, y = d["x_full"], d["y"]
    rev_xf = d["rev_x_full"] if len(d["rev_x_full"]) else None
    rev_y = d["rev_y"] if len(d["rev_y"]) else None
    origin = float(d["origin_m"][0])
    const_gr = d["const_gr"]

    n = len(xf) - 80
    u = np.arange(-30, n + 30.0)
    z_idx = np.arange(n, dtype=float)
    d_xc = float(correlation(xf[40 : 40 + n], y, u, 20.0))

    mp_id = z_idx.copy()
    mp_xc = z_idx + d_xc
    mp_dtw, _ = slope_dtw(xf, y, "local_normalise", band=20.0)
    mp_rc, _ = rcorr_length_map(xf, y, band=20.0)

    g0 = Graph(n, 0.25, 20, 20.0, 0.01, 40, 30)
    sol_b0 = g0.solve(xf, y, MODELS[MODELS[:, 0] == 0])
    sol_bra = g0.solve(xf, y, MODELS)

    mp_r0_snr, *_ = solve_core(xf, y, band=20.0, r0_only=True, adaptive=True)
    res_nara = align_nara(xf, y, band=20.0, rev_xf=rev_xf, rev_y=rev_y, use_gate=True)

    methods = ["IDENTITY", "XCORR", "DTW", "RCORR", "Base-0", "Base-RA", "w/o Broadening", "NARA"]
    maps = np.array([mp_id, mp_xc, mp_dtw, mp_rc, sol_b0["mapped"], sol_bra["mapped"], mp_r0_snr, res_nara["mapped"]])
    wsf_m = origin + maps * 0.1524

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

        eps.append({
            "cohort": wrec["cohort"],
            "well": wrec["well"],
            "case_id": ch["case_id"],
            "channel": ch["channel"],
            "common_n": int(common.sum()),
            "screened_n": int(screened.sum()),
            "lam_eff": round(res_nara["lam_eff"], 4),
            "gate_triggered": bool(res_nara["gate_triggered"]),
            "scores": score_metrics(ref_v, aligned, common, screened, methods),
        })
    return eps


def _well_equal_mean(eps: list, method: str, mask_key: str) -> float:
    wells = sorted(set(e["well"] for e in eps))
    return float(np.mean([
        np.mean([e["scores"][method][mask_key]["rmse"] / e["scores"]["XCORR"][mask_key]["rmse"] for e in eps if e["well"] == w])
        for w in wells
    ]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true", help="Assert exact parity against results/reference_e1_field_13wells.json")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    manifest = json.loads((REPO / "data/field_iodp_13wells/cohort_manifest.json").read_text(encoding="utf-8"))
    t0 = time.perf_counter()
    all_ep = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for res in pool.map(eval_well, manifest):
            all_ep.extend(res)
    print(f"Evaluated {len(manifest)} wells ({len(all_ep)} endpoints) in {time.perf_counter() - t0:.1f} s\n")

    sub_b = [e for e in all_ep if e["cohort"] == "B_eval"]
    for m in ("DTW", "RCORR", "Base-0", "Base-RA", "NARA"):
        r_all = np.array([e["scores"][m]["all"]["rmse"] / e["scores"]["XCORR"]["all"]["rmse"] for e in all_ep])
        r_bf = np.array([e["scores"][m]["all"]["rmse"] / e["scores"]["XCORR"]["all"]["rmse"] for e in sub_b])
        r_bh = np.array([e["scores"][m]["healthy"]["rmse"] / e["scores"]["XCORR"]["healthy"]["rmse"] for e in sub_b])
        we_all = _well_equal_mean(all_ep, m, "all")
        we_bh = _well_equal_mean(sub_b, m, "healthy")
        print(
            f"{m:<8} | All19: {int(np.sum(r_all < 1 - 1e-12)):2d}/19 (>5%: {int(np.sum(r_all > 1.05)):2d}, ep={r_all.mean():.4f}, well={we_all:.4f}) | "
            f"Blind-Full: {int(np.sum(r_bf < 1 - 1e-12)):2d}/15 (ep={r_bf.mean():.4f}) | "
            f"Blind-Screened: {int(np.sum(r_bh < 1 - 1e-12)):2d}/15 (ep={r_bh.mean():.4f}, well={we_bh:.4f})"
        )

    sub_no_aplc = [e for e in all_ep if e["well"] != "U1513E"]
    r_no_aplc = np.array([e["scores"]["NARA"]["all"]["rmse"] / e["scores"]["XCORR"]["all"]["rmse"] for e in sub_no_aplc])
    print(f"\nU1513E-excluded sensitivity (18 endpoints / 12 wells): NARA ep_mean = {r_no_aplc.mean():.4f}")

    if args.verify:
        ref = json.loads((REPO / "results/reference_e1_field_13wells.json").read_text(encoding="utf-8"))
        ref_map = {r["case_id"]: r for r in ref}
        key_map = {
            "IDENTITY": "IDENTITY",
            "XCORR": "XCORR",
            "DTW": "DTW_SLOPE",
            "RCORR": "RCORR_LENGTH",
            "Base-0": "BASE_R0",
            "Base-RA": "BASE_RA",
            "w/o Broadening": "R0_SNR_HUB",
            "NARA": "RA_SNR_HUB_NRGATE",
        }
        max_rmse_diff = 0.0
        for e in all_ep:
            rr = ref_map[e["case_id"]]
            assert e["common_n"] == rr["common_n"], f"{e['case_id']} common_n mismatch"
            assert e["screened_n"] == rr["healthy_n"], f"{e['case_id']} screened_n mismatch"
            for m_new, m_ref in key_map.items():
                for msk in ("all", "healthy"):
                    d_rmse = abs(e["scores"][m_new][msk]["rmse"] - rr["scores"][m_ref][msk]["rmse"])
                    max_rmse_diff = max(max_rmse_diff, d_rmse)
        assert max_rmse_diff <= 1e-6, f"RMSE parity exceeded tolerance: {max_rmse_diff:.3e}"
        print(f"Verification PASSED: max RMSE diff across all 7 methods x 19 endpoints = {max_rmse_diff:.3e}")


if __name__ == "__main__":
    main()
