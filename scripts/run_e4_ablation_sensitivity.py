"""Reproduce Experiment E4 summary tables (Tables 6 & 7: NARA framework ablations & 4-axis sensitivity)."""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def main():
    e4 = json.loads((REPO / "results/reference_e4_ablation_sensitivity.json").read_text(encoding="utf-8"))
    s = e4["summary"]

    print("=== Table 6: NARA Framework Component Ablations ===")
    rows = [
        ("Base-RA (L2, lam=0.01)", "ABL_Base-RA"),
        ("w/o Adapt (Fixed lam=0.05)", "ABL_RA-Fix0.05-Hub"),
        ("w/o Adapt (Fixed lam=2.00)", "ABL_RA-Fix2.0-Hub"),
        ("w/o Huber (L2 Loss)", "ABL_RA-SNR-L2"),
        ("w/o Response (eta=0,1,0)", "ABL_R0-SNR-Hub"),
        ("w/o Gate (Un-gated DP)", "ABL_RA-SNR-Hub"),
        ("NARA (Full Framework)", "ABL_RA-SNR-Hub+Gate"),
    ]
    for name, key in rows:
        r = s[key]
        print(
            f"{name:<28} | Prim={r['primary216_mae']:.4f} | Box={r['box54_mae']:.4f} | "
            f"All19={r['all19_wins']}/19 (gt5%={r['all19_gt5']}, mean={r['all19_mean_ratio']:.4f}) | "
            f"B15_Hth={r['blind15_hth_wins']}/15 ({r['blind15_hth_mean_ratio']:.4f})"
        )

    print("\n=== Table 7: Four-Axis Parameter Sensitivity ===")
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
                f"All19={r['all19_wins']}/19 (gt5%={r['all19_gt5']}, mean={r['all19_mean_ratio']:.4f}) | B15_Hth={r['blind15_hth_mean_ratio']:.4f}"
            )


if __name__ == "__main__":
    main()
