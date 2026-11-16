# NARA: Noise-Adaptive Response Alignment for Repeat-Pass Wireline Logs

Reference implementation, benchmark datasets, and evaluation scripts for **Noise-adaptive response alignment for repeat-pass wireline logs** (Jianwen Rui, Nanjing Vocational College of Information Technology).

`NARA` aligns repeat-pass gamma-ray (GR) borehole logs by combining four mechanisms in a slope-bounded dynamic programming solver:

- **45-member response bank**: discrete Gaussian smoothing (`sigma in {0, 1, 2, 3, 4}` samples), multiplicative gain (`a in {0.8, 1.0, 1.2}`), and baseline offset (`b in {-0.2, 0.0, 0.2}`).
- **Huber edge penalty** (`delta = 1.0`): bounds outlier pull from washout zones, rugosity spikes, and casing transitions.
- **Two-pass SNR-adaptive stiffness** (`lam_eff = 0.05 + 15 * J_data`): uses the irreducible Pass-1 residual `J_data` to keep clean benchmarks flexible (`lam_eff ~ 0.14`) while stabilizing noisy borehole runs (`lam_eff = 0.53..9.51`).
- **Two-way reciprocity gate**: reverts to bulk cross-correlation (`XCORR`) when forward and reversed passes both saturate on the same extreme gain bound (`0.8` or `1.2`).

Exact forward-backward dynamic programming yields both the optimal depth map and whole-interval conditional cost profiles `Q_k(v)`.

## Contents

| Path | Description |
|---|---|
| `nara/` | Core solver package (`graph.py`, `solver.py`, `__init__.py`) |
| `data/synthetic_1032/` | Pre-built synthetic benchmark cases and index (`manifest.json`, 1,032 cases) |
| `data/field_iodp_13wells/` | Extracted GR passes, held-out channels, and masks for 13 IODP holes (19 endpoints) |
| `data/iodp_dlis_manifest.json` | Upstream LDEO IODP-USIO filenames, portal link, and SHA-256 checksums |
| `scripts/` | CLI scripts reproducing quickstart, field, synthetic, conditional-QA, ablation, and figure outputs |
| `results/` | Saved JSON outputs and plotted curve arrays matching the manuscript tables |
| `figures/` | Exported vector figures (`PDF` and `SVG`) plus PNG previews |

## Installation

Python 3.10 or newer with NumPy, SciPy, and Matplotlib:

```bash
pip install -r requirements.txt
pip install -e .
```

## Basic Usage

```python
import numpy as np
from nara import align_nara

data = np.load("data/field_iodp_13wells/U1611A.npz")
res = align_nara(
    data["x_full"],
    data["y"],
    band=20.0,
    rev_xf=data["rev_x_full"],
    rev_y=data["rev_y"],
    profile=True,
)
print(res["lam_eff"], res["model"], res["gate_triggered"])
```

Run the bundled two-well demo (`U1611A` and `U1437D`):

```bash
python scripts/quickstart_example.py
```

## Reproducing Paper Results

```bash
python scripts/run_e1_field_13wells.py --verify
python scripts/run_e2_synthetic_1032.py
python scripts/run_e3_conditional_qa.py
python scripts/run_e4_ablation_sensitivity.py
python scripts/make_figures.py
```

## Data Notes

- Synthetic cases (`data/synthetic_1032/`) originate from 9 public SPWLA wells (`0.1524 m` sampling; Third SPWLA Machine Learning Competition, 2023) plus 3 U1580A borehole windows.
- Field logs originate from the LDEO IODP-USIO wireline archive (<https://mlp.ldeo.columbia.edu/data/iodp-usio/>). Raw DLIS files reach up to 401 MB per file and are omitted here; pre-extracted depth-paired arrays and quality masks for all 13 holes live in `data/field_iodp_13wells/`, with original filenames and SHA-256 digests recorded in `data/iodp_dlis_manifest.json`.

## License

MIT License (see `LICENSE`).
