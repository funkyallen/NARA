"""NARA solver: Huber edge loss, two-pass pilot-scaled stiffness, reciprocity check, and helpers."""

from __future__ import annotations

import numpy as np
from .graph import MODELS, Graph, correlation, scale_pair


def huber_loss(res: np.ndarray, delta: float = 1.0) -> np.ndarray:
    """Pointwise Huber loss (or 0.5 * squared error matching quadratic core when delta is infinite)."""
    if not np.isfinite(delta):
        return 0.5 * res * res
    ar = np.abs(res)
    # Linear tail above 1 MAD unit keeps washouts and spikes from dragging the warp
    return np.where(ar <= delta, 0.5 * res * res, delta * (ar - 0.5 * delta))


def compute_base_costs(
    g: Graph,
    xf: np.ndarray,
    y: np.ndarray,
    models: np.ndarray = MODELS,
    delta: float = 1.0,
    templates: np.ndarray | None = None,
):
    """Precompute segment data costs D_e (independent of lambda) for two-pass reuse."""
    if templates is None:
        templates = g.templates(xf, models)
    base_costs = []
    for e in range(g.K - 1):
        lo, hi = g.knots[e : e + 2]
        h = hi - lo
        pre, valid = g.edges[e]
        t = np.arange(h + 1) / h
        z_seg = np.arange(lo, hi + 1, dtype=float)
        mapped = z_seg[None, None, :] + g.grid[pre, None] * (1.0 - t) + g.grid[:, None, None] * t
        obs = np.interp(mapped, g.u, y)
        res = obs[None, :, :, :] - templates[:, None, None, lo : hi + 1]
        weights = np.ones(h + 1)
        weights[[0, -1]] = 0.5
        loss = huber_loss(res, delta=delta)
        cost = np.sum(loss * weights, axis=-1) / (g.n - 1)
        cost[:, ~valid] = np.inf
        base_costs.append(cost)
    return templates, base_costs


def dp_with_precomputed(
    g: Graph,
    xf: np.ndarray,
    y: np.ndarray,
    models: np.ndarray,
    templates: np.ndarray,
    base_costs,
    lam: float,
    profile: bool = False,
):
    """Solve exact forward (and optional backward) DP reusing precomputed data edge tables."""
    costs = []
    for e, bc in enumerate(base_costs):
        lo, hi = g.knots[e : e + 2]
        h = hi - lo
        pre, valid = g.edges[e]
        pen = np.where(valid, lam * (g.grid[:, None] - g.grid[pre]) ** 2 / (h * (g.n - 1)), np.inf)
        c = bc + pen[None, :, :]
        c[:, ~valid] = np.inf
        costs.append(c)
    return g.solve(xf, y, models=models, costs=costs, templates=templates, profile=profile)


def pilot_data_residual(g: Graph, base_costs, sol1: dict) -> float:
    """Compute normalised irreducible pilot data residual J_data from Pass-1 path."""
    p = sol1["path"]
    m = int(sol1["model_index"])
    j_data = 0.0
    for e in range(g.K - 1):
        pre, valid = g.edges[e]
        j_idx = int(np.flatnonzero(valid[p[e + 1]] & (pre[p[e + 1]] == p[e]))[0])
        j_data += float(base_costs[e][m, p[e + 1], j_idx])
    return j_data


def is_same_side_bound_sat(mdl_fwd, mdl_rev) -> bool:
    """Flag physically inconsistent fits where both directions saturate on a=0.8 or a=1.2."""
    if mdl_rev is None:
        return False
    a_f, a_r = float(mdl_fwd[1]), float(mdl_rev[1])
    return (abs(a_f - 0.8) < 1e-6 and abs(a_r - 0.8) < 1e-6) or (
        abs(a_f - 1.2) < 1e-6 and abs(a_r - 1.2) < 1e-6
    )


def solve_core(
    xf: np.ndarray,
    y: np.ndarray,
    band: float = 20.0,
    *,
    delta: float = 1.0,
    lam0: float = 0.05,
    kappa: float = 15.0,
    knot_step: int = 20,
    grid_step: float = 0.25,
    r0_only: bool = False,
    adaptive: bool = True,
    profile: bool = False,
):
    """Run two-pass Huber response-aware DP (supports ablations via flags)."""
    n = len(xf) - 80
    g = Graph(n, grid_step, knot_step, band, 0.0, 40, 30)
    models = MODELS[MODELS[:, 0] == 0] if r0_only else MODELS
    templates, base_costs = compute_base_costs(g, xf, y, models, delta=delta)
    if not adaptive:
        sol = dp_with_precomputed(g, xf, y, models, templates, base_costs, lam0, profile=profile)
        return sol["mapped"], sol["model"].tolist(), float(lam0), 0.0, sol, g, templates, base_costs
    sol1 = dp_with_precomputed(g, xf, y, models, templates, base_costs, lam0, profile=False)
    j_data = pilot_data_residual(g, base_costs, sol1)
    lam_eff = lam0 + kappa * j_data
    sol2 = dp_with_precomputed(g, xf, y, models, templates, base_costs, lam_eff, profile=profile)
    return sol2["mapped"], sol2["model"].tolist(), float(lam_eff), float(j_data), sol2, g, templates, base_costs


def align_nara(
    xf: np.ndarray,
    y: np.ndarray,
    band: float = 20.0,
    *,
    rev_xf: np.ndarray | None = None,
    rev_y: np.ndarray | None = None,
    delta: float = 1.0,
    lam0: float = 0.05,
    kappa: float = 15.0,
    knot_step: int = 20,
    grid_step: float = 0.25,
    use_gate: bool = True,
    profile: bool = False,
) -> dict:
    """Align a scaled curve pair with Noise-Adaptive Response Alignment (NARA)."""
    mp_fwd, mdl_fwd, lam_eff, j_data, sol2, g, templates, base_costs = solve_core(
        xf,
        y,
        band=band,
        delta=delta,
        lam0=lam0,
        kappa=kappa,
        knot_step=knot_step,
        grid_step=grid_step,
        r0_only=False,
        adaptive=True,
        profile=profile,
    )

    mdl_rev = None
    gate_triggered = False
    if use_gate and rev_xf is not None and rev_y is not None:
        _, mdl_rev, _, _, _, _, _, _ = solve_core(
            rev_xf,
            rev_y,
            band=band,
            delta=delta,
            lam0=lam0,
            kappa=kappa,
            knot_step=knot_step,
            grid_step=grid_step,
            r0_only=False,
            adaptive=True,
            profile=False,
        )
        gate_triggered = is_same_side_bound_sat(mdl_fwd, mdl_rev)

    if gate_triggered:
        n = len(xf) - 80
        u = np.arange(-30, n + 30.0)
        d_xc = float(correlation(xf[40 : 40 + n], y, u, band))
        mapped = np.arange(n, dtype=float) + d_xc
    else:
        mapped = mp_fwd

    return {
        "mapped": mapped,
        "mapped_dp": mp_fwd,
        "model": mdl_fwd,
        "rev_model": mdl_rev,
        "lam_eff": lam_eff,
        "j_data_pilot": j_data,
        "gate_triggered": gate_triggered,
        "solution": sol2,
        "graph": g,
        "templates": templates,
        "base_costs": base_costs,
    }


def align_raw_curves(
    ref_depth: np.ndarray,
    ref_gr: np.ndarray,
    tar_depth: np.ndarray,
    tar_gr: np.ndarray,
    *,
    band_m: float = 3.048,
    step_m: float | None = None,
    use_gate: bool = True,
    profile: bool = True,
) -> dict:
    """Register two raw gamma-ray passes in native depth and gAPI units.

    Resamples onto a regular overlapping depth grid, constructs 40-sample reference
    halos and 30-sample target margins, applies `scale_pair` MAD normalisation,
    runs two-pass Huber DP with optional two-way reciprocity check, and returns
    aligned target depths, shift offsets in metres, and Q_k(v) conditioning width.
    """
    rd, rv = np.asarray(ref_depth, dtype=float), np.asarray(ref_gr, dtype=float)
    td, tv = np.asarray(tar_depth, dtype=float), np.asarray(tar_gr, dtype=float)
    rmask, tmask = good(rd) & good(rv), good(td) & good(tv)
    rd, rv, td, tv = rd[rmask], rv[rmask], td[tmask], tv[tmask]
    if step_m is None:
        step_m = float(np.median(np.diff(rd)))
    band_samples = max(4.0, round(band_m / step_m))
    lo = max(float(rd[0]), float(td[0])) + 40.0 * step_m
    hi = min(float(rd[-1]), float(td[-1])) - 40.0 * step_m
    if hi <= lo + 20.0 * step_m:
        raise ValueError("overlapping depth interval is too short for 40-sample halos")

    z_core = np.arange(lo, hi + 0.25 * step_m, step_m)
    n = len(z_core)
    z_xf = lo + np.arange(-40, n + 40, dtype=float) * step_m
    z_y = lo + np.arange(-30, n + 30, dtype=float) * step_m
    xf_raw = np.interp(z_xf, rd, rv)
    y_raw = np.interp(z_y, td, tv)
    xf, y, scale_info = scale_pair(xf_raw, y_raw, margin=40)

    rev_xf, rev_y = None, None
    if use_gate:
        rev_xf_raw = np.interp(z_xf, td, tv)
        rev_y_raw = np.interp(z_y, rd, rv)
        rev_xf, rev_y, _ = scale_pair(rev_xf_raw, rev_y_raw, margin=40)

    res = align_nara(xf, y, band=band_samples, rev_xf=rev_xf, rev_y=rev_y, use_gate=use_gate, profile=profile)
    shift_m = (res["mapped"] - np.arange(n, dtype=float)) * step_m
    aligned_tar_gr = np.interp(z_core + shift_m, td, tv)

    width_m = None
    norm_width_m = None
    if profile:
        g = res["graph"]
        q = res["solution"]["q"]
        dq = q[g.K // 2].min(axis=0) - res["solution"]["value"]
        sub = g.grid[dq <= 0.005] * step_m
        width_m = float(sub[-1] - sub[0]) if len(sub) else 0.0
        sub_norm = g.grid[dq <= 0.005 * res["lam_eff"]] * step_m
        norm_width_m = float(sub_norm[-1] - sub_norm[0]) if len(sub_norm) else 0.0

    return {
        "depth_m": z_core,
        "ref_gr": xf_raw[40 : 40 + n],
        "aligned_tar_depth_m": z_core + shift_m,
        "aligned_tar_gr": aligned_tar_gr,
        "shift_m": shift_m,
        "step_m": step_m,
        "lam_eff": res["lam_eff"],
        "j_data_pilot": res["j_data_pilot"],
        "model": res["model"],
        "rev_model": res["rev_model"],
        "gate_triggered": res["gate_triggered"],
        "mid_width_005_m": width_m,
        "mid_width_norm_m": norm_width_m,
        "scale_info": scale_info,
    }


def eval_continuous_obj(g: Graph, xf: np.ndarray, y: np.ndarray, model, offsets: np.ndarray, lam_eff: float, delta: float = 1.0) -> float:
    """Direct continuous trapezoidal + smoothness objective evaluation on arbitrary offsets."""
    mapped = g.z + np.interp(g.z, g.knots, offsets)
    tpl = g.templates(xf, np.array([model]))[0]
    res = np.interp(mapped, g.u, y) - tpl
    w = np.ones(g.n)
    w[[0, -1]] = 0.5
    data_term = float(np.sum(huber_loss(res, delta=delta) * w) / (g.n - 1))
    reg_term = float(lam_eff * np.sum(np.diff(offsets) ** 2 / g.h) / (g.n - 1))
    return data_term + reg_term


def alternating_huber_dp(g: Graph, templates: np.ndarray, base_costs, starts, lam_eff: float, xf: np.ndarray, y: np.ndarray, models: np.ndarray = MODELS, delta: float = 1.0):
    """8-start alternating response-and-alignment solver (ALT8) on the Pass-2 NARA objective."""
    cache = {}
    runs_out = []
    for start in starts:
        offsets = np.asarray(start, dtype=float).copy()
        mi = 0
        for _ in range(50):
            vals = np.array([eval_continuous_obj(g, xf, y, models[i], offsets, lam_eff, delta=delta) for i in range(len(models))])
            mi = int(vals.argmin())
            if mi not in cache:
                cache[mi] = dp_with_precomputed(
                    g, xf, y, models[mi : mi + 1], templates[mi : mi + 1], [bc[mi : mi + 1] for bc in base_costs], lam_eff, profile=False
                )
            sol = cache[mi]
            new_off = g.grid[sol["path"]]
            if np.array_equal(offsets, new_off) or vals[mi] - sol["value"] <= 1e-12:
                offsets = new_off
                break
            offsets = new_off
        runs_out.append({"offsets": offsets, "model_index": mi, "value": float(cache[mi]["value"])})
    best = min(runs_out, key=lambda r: r["value"])
    return {
        **best,
        "model": models[best["model_index"]].tolist(),
        "mapped": g.z + np.interp(g.z, g.knots, best["offsets"]),
    }


def good(a: np.ndarray) -> np.ndarray:
    """Filter non-finite values and DLIS null sentinels while keeping negative values (e.g., log resistivity)."""
    return np.isfinite(a) & (np.abs(a) < 1e10) & ~np.isin(a, [-999.25, -9999.25, -999.0, -9999.0])


def constant_runs_mask(v: np.ndarray, tol: float = 1e-7, min_len: int = 5) -> np.ndarray:
    """Flag flat-tail or stuck-tool constant runs of length >= min_len."""
    out = np.zeros(len(v), dtype=bool)
    i = 0
    while i < len(v):
        if not np.isfinite(v[i]):
            i += 1
            continue
        j = i + 1
        while j < len(v) and np.isfinite(v[j]) and abs(v[j] - v[i]) <= tol:
            j += 1
        if j - i >= min_len:
            out[i:j] = True
        i = j
    return out


def sample_aux(depth_src: np.ndarray, val_src: np.ndarray, at: np.ndarray, step: float, pos_only: bool) -> np.ndarray:
    """Linearly interpolate a held-out evaluation channel without spanning gaps or null values."""
    hi = np.clip(np.searchsorted(depth_src, at, side="right"), 1, len(depth_src) - 1)
    lo = hi - 1
    valid = (
        (at >= depth_src[0] - 1e-8)
        & (at <= depth_src[-1] + 1e-8)
        & good(val_src[lo])
        & good(val_src[hi])
        & (depth_src[hi] - depth_src[lo] <= step + 1e-7)
    )
    if pos_only:
        valid &= (val_src[lo] > 0) & (val_src[hi] > 0)
    out = val_src[lo] + (val_src[hi] - val_src[lo]) * (at - depth_src[lo]) / (depth_src[hi] - depth_src[lo])
    out[~valid] = np.nan
    return out


def score_metrics(ref_v: np.ndarray, aligned_arr: np.ndarray, common: np.ndarray, healthy: np.ndarray, methods: list[str]) -> dict:
    """Compute All-Mask and Healthy-Mask RMSE, MAE, and Pearson r across methods."""
    out = {}
    for mi, m in enumerate(methods):
        def _stats(mask):
            a = ref_v[mask]
            b = aligned_arr[mi][mask]
            e = b - a
            aa = a - a.mean()
            bb = b - b.mean()
            den = float(np.sqrt(np.dot(aa, aa) * np.dot(bb, bb)))
            r = float(np.dot(aa, bb) / den) if den > 0 else 0.0
            return {"rmse": float(np.sqrt(np.mean(e * e))), "mae": float(np.mean(np.abs(e))), "r": r}

        out[m] = {"all": _stats(common), "healthy": _stats(healthy) if healthy.any() else None}
    return out
