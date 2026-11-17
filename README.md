# NARA: Noise-Adaptive Response Alignment for Repeat-Pass Wireline Logs

Reference implementation, raw-curve alignment CLI, benchmark datasets, and verification scripts for **Noise-adaptive response alignment for repeat-pass wireline logs** (Jianwen Rui, Nanjing Vocational College of Information Technology).

`NARA` aligns repeat-pass gamma-ray (GR) borehole logs by combining four mechanisms inside an exact slope-bounded forward-backward dynamic programming solver:

- **45-member discrete response bank**: centred Gaussian broadening (`sigma in {0, 1, 2, 3, 4}` samples), multiplicative gain (`a in {0.8, 1.0, 1.2}`), and baseline offset (`b in {-0.2, 0.0, 0.2}`).
- **Huber edge penalty** (`delta = 1.0`): caps outlier leverage from borehole washouts, rugosity spikes, and casing transitions while matching quadratic `0.5 * e^2` behaviour inside `|e| <= 1`.
- **Two-pass residual-adaptive stiffness** (`lam_eff = 0.05 + 15 * J_data`): scales path stiffness linearly with the Pass-1 Huber residual `J_data`, keeping clean broadened traces flexible (`lam_eff ~ 0.14`) while stabilising noisy field logs (`lam_eff = 0.53..9.51`).
- **Two-way reciprocity safeguard**: runs the reversed direction (`y -> x`) when paired directional passes exist and reverts to bulk cross-correlation (`XCORR`) when both passes lock onto the same gain bound (`a_fwd = a_rev in {0.8, 1.2}`).

Exact forward-backward DP simultaneously outputs deterministic whole-interval excess-cost profiles `Q_k(v)` and witness trajectories without random restarts.

## Contents

| Path | Description |
|---|---|
| `nara/` | Core package (`graph.py`, `solver.py`, `cli.py`, `__init__.py`) including `align_raw_curves` and `scale_pair` |
| `data/synthetic_1032/` | Pre-built synthetic benchmark cases (`manifest.json`, 1,032 cases) |
| `data/field_iodp_13wells/` | Extracted GR passes, held-out channels, and screened masks for 13 IODP boreholes (19 endpoints) |
| `data/iodp_dlis_manifest.json` | Upstream LDEO IODP-USIO DLIS filenames, portal links, and SHA-256 digests |
| `scripts/` | Verification scripts (`run_e1..e4`) and figure builder (`make_figures.py`) |
| `results/` | Saved JSON archives (`reference_*.json`) and plotting arrays (`fig_cache.npz`) |
| `figures/` | Exported vector figures (`PDF` and `SVG`) plus PNG previews |

## Installation

Python 3.10 or newer with NumPy, SciPy, and Matplotlib:

```bash
pip install -r requirements.txt
pip install -e .
```

## Using NARA on Your Own Repeat-Pass Logs

### Option A: Raw CSV Command-Line Interface (`nara-align`)

Pass any two depth-indexed gamma-ray CSV files (`depth_m, gr`) in native metre and `gAPI` units. The CLI resamples onto the common grid, adds the required 40-sample reference halos and 30-sample target margins, applies shared-reference `scale_pair` MAD normalisation, executes NARA with reciprocity checking, and exports aligned depths plus `Q_k(v)` diagnostics:

```bash
nara-align --ref examples/u1445a_ref_gr.csv --tar examples/u1445a_tar_gr.csv --out u1445a_aligned.csv --band-m 3.048
```

Verified output on the bundled raw `U1445A` sample (`gAPI` scale):

```json
{
  "n_core": 1940,
  "step_m": 0.1524000000000001,
  "lam_eff": 1.7353,
  "j_data_pilot": 0.11235,
  "model_sigma_a_b": [2.0, 1.0, 0.0],
  "rev_model_sigma_a_b": [2.0, 1.0, -0.2],
  "gate_triggered": false,
  "mid_width_005_m": 3.657600000000002,
  "mid_width_norm_m": 5.334000000000003
}
```

Aligned CSV columns (`u1445a_aligned.csv`): `depth_m,ref_gr,aligned_tar_depth_m,shift_m,aligned_tar_gr`.

### Option B: Python API (`align_raw_curves` vs Core `align_nara`)

For raw curves in physical units, call `align_raw_curves` directly:

```python
from nara import align_raw_curves

res = align_raw_curves(ref_depth_m, ref_gr, tar_depth_m, tar_gr, band_m=3.048, use_gate=True, profile=True)
print(res["lam_eff"], res["model"], res["gate_triggered"], res["mid_width_005_m"])
```

If calling the low-level array solver `align_nara(x_full, y, ...)` directly, `x_full` (`len = n + 80`, including 40-sample edge halos on each side) and `y` (`len = n + 60`, including 30-sample search margins) **must** first be normalised jointly using `scale_pair(x_full, y, margin=40)` (`1.4826 * MAD` of the reference core). Pre-extracted arrays inside `data/field_iodp_13wells/` already satisfy this contract:

```python
import numpy as np
from nara import align_nara

d = np.load("data/field_iodp_13wells/U1611A.npz")
res = align_nara(d["x_full"], d["y"], band=20.0, rev_xf=d["rev_x_full"], rev_y=d["rev_y"], profile=True)
print(res["lam_eff"], res["model"], res["gate_triggered"])
```

Run the bundled two-well demonstration (`U1611A` and `U1437D`):

```bash
python scripts/quickstart_example.py
```

## Reproducing Paper Results

```bash
python scripts/run_e1_field_13wells.py --verify
python scripts/run_e2_synthetic_1032.py --with-ablations --verify
python scripts/run_e3_conditional_qa.py --with-synthetic --verify
python scripts/run_e4_ablation_sensitivity.py --recompute-ablations
python scripts/make_figures.py
```

Execution scope matrix:

| Script | What Is Computed Live from `data/` | What Uses `results/` Archives |
|---|---|---|
| `run_e1_field_13wells.py --verify` | Full 13-well / 19-endpoint run across 8 comparators + well-equal means + `U1513E`-excluded check | Asserts `<= 1e-6` RMSE parity against `reference_e1_field_13wells.json` |
| `run_e2_synthetic_1032.py --with-ablations --verify` | All 1,032 synthetic cases (`Primary Dev 01-06` vs `Held-out 07-09`, `sigma=0` over `n=216`, optional `Base-RA` & `w/o Broadening` bootstrap) | Asserts `<= 1e-6` well-equal MAE parity against `reference_e2_synthetic_1032.json` |
| `run_e3_conditional_qa.py --with-synthetic --verify` | Full `Q_k(v)` witness-path parity and deterministic non-oracle 8-start `ALT8` across 6 field holes and 90 synthetic tests | Uses synthetic case IDs and field assertion targets in `reference_e3_conditional_qa.json` |
| `run_e4_ablation_sensitivity.py --recompute-ablations` | Table 6 field variants (`ABL_Base-RA`, `Fix0.05`, `Fix2.00`, true `0.5*e^2` `w/o Huber`, `w/o Broadening`, `w/o Gate`, `NARA`) | Default mode prints saved Table 6 and Table 7 (`SEN_*`) from `reference_e4_ablation_sensitivity.json` |
| `make_figures.py` | Renders all vector PDF/SVG/PNG plots (`Figures 1-7`) | Reads curve caches and metrics from `results/` |

Note on `nara.solver.huber_loss(..., delta=float("inf"))`: when `delta` is non-finite, `huber_loss` returns `0.5 * res * res` so quadratic ablations (`w/o Huber`) share the exact quadratic-core scaling of Huber loss (`0.5 * e^2` for `|e| <= delta`). Note also that `kappa = 15` and `Delta Q <= 0.005` thresholds calibrate to the common `0.1524 m` sample spacing and `h = 20` knot step.

## Data Notes

- Synthetic cases (`data/synthetic_1032/`) originate from 9 public SPWLA wells (`0.1524 m` sampling; Third SPWLA Machine Learning Competition, 2023) plus 3 `U1580A` borehole windows.
- Field logs originate from the LDEO IODP-USIO wireline archive (<https://mlp.ldeo.columbia.edu/data/iodp-usio/>). Original DLIS files reach up to `401 MB` per file and are omitted here; pre-extracted depth-paired arrays and screened quality masks for all 13 boreholes reside in `data/field_iodp_13wells/`, with source filenames and SHA-256 digests in `data/iodp_dlis_manifest.json`. Note that `RLA5` is scored primarily in linear `ohm.m` (with `log10(RLA5)` sensitivity reported in the manuscript), and `U1513E` `APLC` spans `0.2241..7.4460 m3/m3` under poor hole conditions.

## License

MIT License (see `LICENSE`).
