"""Slope-bounded dynamic programming graph, 45-member response bank, and baselines."""

from __future__ import annotations

import itertools
import numpy as np
from scipy.ndimage import gaussian_filter1d, uniform_filter1d
from scipy.optimize import minimize
from scipy.interpolate import PchipInterpolator
from scipy.signal import savgol_filter

# 45-member discrete relative-response bank: (sigma, gain a, baseline b)
MODELS = np.array(
    list(itertools.product([0.0, 1.0, 2.0, 3.0, 4.0], [0.8, 1.0, 1.2], [-0.2, 0.0, 0.2])),
    dtype=float,
)


def blur(x: np.ndarray, sigma: float) -> np.ndarray:
    """Apply centred Gaussian resolution broadening in sample units."""
    return x.copy() if sigma == 0 else gaussian_filter1d(x, sigma, mode="reflect", truncate=4.0)


def scale_pair(xf: np.ndarray, y: np.ndarray, margin: int = 40):
    """Scale reference and target curves using reference-core median and 1.4826 * MAD."""
    core = xf[margin : len(xf) - margin] if margin else xf
    med = float(np.median(core))
    scale = float(1.4826 * np.median(np.abs(core - med)))
    if not np.isfinite(scale) or scale <= 1e-7:
        raise ValueError("insufficient_reference_GR_scale")
    return (xf - med) / scale, (y - med) / scale, {"center": med, "scale": scale}


def robust_scale(v: np.ndarray, center: np.ndarray | None = None) -> np.ndarray:
    """Normalise a curve by median and robust MAD spread."""
    ref = v if center is None else center
    med = np.median(ref)
    s = 1.4826 * np.median(np.abs(ref - med))
    return (v - med) / max(s, 1e-7)


class Graph:
    """Discrete slope-bounded piecewise-linear warping graph with exact forward-backward DP."""

    def __init__(
        self,
        n: int,
        step: float = 0.25,
        interval: int = 20,
        band: float = 20.0,
        lam: float = 0.05,
        margin: int = 40,
        y_margin: int = 30,
    ):
        self.n = int(n)
        self.z = np.arange(self.n, dtype=float)
        self.u = np.arange(-y_margin, self.n + y_margin, dtype=float)
        self.grid = np.arange(round(2 * band / step) + 1) * step - band
        self.knots = np.r_[np.arange(0, self.n - 1, interval), self.n - 1].astype(int)
        self.h = np.diff(self.knots)
        self.step = float(step)
        self.band = float(band)
        self.lam = float(lam)
        self.margin = int(margin)
        self.S = len(self.grid)
        self.K = len(self.knots)
        self.edges = []
        for h in self.h:
            # Cap knot-to-knot offset change so segment stretch stays in [0.8, 1.2]
            reach = int(np.floor(0.2 * h / step + 1e-8))
            delta = np.arange(-reach, reach + 1)
            pre = np.arange(self.S)[:, None] + delta[None, :]
            valid = (pre >= 0) & (pre < self.S)
            self.edges.append((np.clip(pre, 0, self.S - 1), valid))

    def templates(self, xf: np.ndarray, models: np.ndarray = MODELS) -> np.ndarray:
        assert len(xf) == self.n + 2 * self.margin
        responses = {s: blur(xf, s)[self.margin : self.margin + self.n] for s in set(models[:, 0])}
        return np.array([a * responses[s] + b for s, a, b in models])

    def edge(self, templates: np.ndarray, y: np.ndarray, e: int, lam: float | None = None) -> np.ndarray:
        lo, hi = self.knots[e : e + 2]
        h = hi - lo
        pre, valid = self.edges[e]
        t = np.arange(h + 1) / h
        z = np.arange(lo, hi + 1, dtype=float)
        mapped = z[None, None, :] + self.grid[pre, None] * (1.0 - t) + self.grid[:, None, None] * t
        obs = np.interp(mapped, self.u, y)
        residual = obs[None, :, :, :] - templates[:, None, None, lo : hi + 1]
        weights = np.ones(h + 1)
        weights[[0, -1]] = 0.5  # Trapezoidal integration weights over the segment
        cost = np.sum(residual * residual * weights, axis=-1) / (self.n - 1)
        penalty = (self.lam if lam is None else lam) * (self.grid[:, None] - self.grid[pre]) ** 2 / (h * (self.n - 1))
        cost += penalty[None, :, :]
        cost[:, ~valid] = np.inf
        return cost

    def solve(
        self,
        xf: np.ndarray,
        y: np.ndarray,
        models: np.ndarray = MODELS,
        profile: bool = False,
        costs=None,
        templates: np.ndarray | None = None,
    ):
        models = np.asarray(models)
        M = len(models)
        if templates is None:
            templates = self.templates(xf, models)
        f = np.zeros((M, self.S))
        parents = np.empty((self.K - 1, M, self.S), dtype=np.int16)
        forwards = [f.copy()] if profile else None
        for e in range(self.K - 1):
            c = self.edge(templates, y, e) if costs is None else costs[e]
            pre, _ = self.edges[e]
            v = f[:, pre] + c
            idx = v.argmin(axis=-1)
            parents[e] = np.take_along_axis(np.broadcast_to(pre, (M, *pre.shape)), idx[:, :, None], axis=-1)[:, :, 0]
            f = np.take_along_axis(v, idx[:, :, None], axis=-1)[:, :, 0]
            if profile:
                forwards.append(f.copy())

        mi, last = np.unravel_index(f.argmin(), f.shape)
        p = self.backtrack(parents, int(mi), int(last))
        result = {
            "path": p,
            "model_index": int(mi),
            "model": models[mi],
            "value": float(f[mi, last]),
            "mapped": self.z + np.interp(self.z, self.knots, self.grid[p]),
        }
        if profile:
            b = np.zeros((M, self.S))
            backs = [b.copy()]
            nexts = np.empty_like(parents)
            for e in range(self.K - 2, -1, -1):
                c = self.edge(templates, y, e) if costs is None else costs[e]
                pre, valid = self.edges[e]
                best = np.full((M, self.S), np.inf)
                nxt = np.zeros((M, self.S), dtype=np.int16)
                for j in range(pre.shape[1]):
                    dest = np.flatnonzero(valid[:, j])
                    src = pre[dest, j]
                    v = c[:, dest, j] + b[:, dest]
                    improve = v < best[:, src]
                    best[:, src] = np.minimum(best[:, src], v)
                    nxt[:, src] = np.where(improve, dest[None, :], nxt[:, src])
                b = best
                backs.append(b.copy())
                nexts[e] = nxt
            # Exact min-cost profile conditioned on knot k passing through grid state v
            q = np.array(forwards) + np.array(backs[::-1])
            result.update(q=q, parents=parents, nexts=nexts)
        return result

    def backtrack(self, parents: np.ndarray, m: int, last: int) -> np.ndarray:
        p = np.empty(self.K, dtype=np.int16)
        p[-1] = last
        for e in range(self.K - 2, -1, -1):
            p[e] = parents[e, m, p[e + 1]]
        return p

    def witness(self, solution: dict, k: int, v: int):
        """Extract exact optimal model, knot path, and objective value pinned at knot k = g_v."""
        m = int(solution["q"][k, :, v].argmin())
        p = np.empty(self.K, dtype=np.int16)
        p[k] = v
        for e in range(k - 1, -1, -1):
            p[e] = solution["parents"][e, m, p[e + 1]]
        for e in range(k, self.K - 1):
            p[e + 1] = solution["nexts"][e, m, p[e]]
        return m, p, float(solution["q"][k, m, v])


def correlation(x: np.ndarray, y: np.ndarray, u: np.ndarray, band: float) -> int:
    """Rigid integer bulk cross-correlation shift (XCORR)."""
    z = np.arange(len(x))
    xx = x - x.mean()
    values = []
    shifts = sorted(range(-int(band), int(band) + 1), key=lambda k: (abs(k), k))
    for k in shifts:
        yy = np.interp(z + k, u, y)
        yy -= yy.mean()
        den = np.linalg.norm(xx) * np.linalg.norm(yy)
        values.append(float(xx @ yy / den) if den > 1e-12 else -1.0)
    return shifts[int(np.argmax(values))]


def _preprocess(xf: np.ndarray, y: np.ndarray, kind: str, margin: int = 40, ymargin: int = 30):
    n = len(xf) - 2 * margin
    if kind == "common_smooth":
        xx = blur(xf, 2)[margin : margin + n]
        yy = blur(y, 2)
        return robust_scale(xx), robust_scale(yy, yy[ymargin : ymargin + n])

    def local(v):
        v = savgol_filter(v, 11, 2)
        mu = uniform_filter1d(v, 41, mode="reflect")
        var = np.maximum(uniform_filter1d(v * v, 41, mode="reflect") - mu * mu, 0.0)
        return (v - mu) / np.maximum(np.sqrt(var), max(1e-7, 1e-6 * np.std(v)))

    return local(xf)[margin : margin + n], local(y)


def slope_dtw(xf: np.ndarray, y: np.ndarray, kind: str = "local_normalise", band: float = 16.0, margin: int = 40, ymargin: int = 30):
    """Slope-constrained dynamic time warping (DTW; Sakoe & Chiba, 1978)."""
    x, yy = _preprocess(xf, y, kind, margin, ymargin)
    n = len(x)
    g = Graph(n, 1.0, 5, band, 0.0, margin, ymargin)
    padded = np.pad(x, (margin, margin), mode="reflect")
    sol = g.solve(padded, yy, np.array([[0.0, 1.0, 0.0]]))
    return sol["mapped"], {"status": "fitted", "preprocessing": kind, "value": sol["value"]}


def regularized_correlation(z, x, u, y, cfg, band, global_shift):
    """Continuous length-scaled regularised correlation (RCORR; Westeng et al., 2026)."""
    knots = np.linspace(z[0], z[-1], cfg["knots"])
    width = np.diff(knots)
    basis = np.column_stack([np.interp(z, knots, column) for column in np.eye(len(knots))])
    a = cfg["magnitude_penalty"] * width / 3.0
    b = cfg["roughness_penalty"] / width
    quadratic = np.diag(np.r_[a + b, 0.0] + np.r_[0.0, a + b])
    quadratic += np.diag(a / 2.0 - b, 1) + np.diag(a / 2.0 - b, -1)
    x = savgol_filter(x, cfg["smooth_window"], cfg["smooth_order"])
    x = x - x.mean()
    norm = np.linalg.norm(x)
    start = np.full(len(knots), global_shift)
    if norm <= 1e-12:
        return z + global_shift, "flat_reference", 0
    x = x / norm
    curve = PchipInterpolator(u, savgol_filter(y, cfg["smooth_window"], cfg["smooth_order"]), extrapolate=False)
    derivative = curve.derivative()

    def objective(offset):
        mapped = np.clip(z + basis @ offset, u[0], u[-1])
        aligned = curve(mapped)
        aligned -= aligned.mean()
        scale = np.linalg.norm(aligned)
        penalty_grad_half = quadratic @ offset
        penalty = offset @ penalty_grad_half
        if scale <= 1e-12:
            return penalty, 2.0 * penalty_grad_half
        aligned /= scale
        corr = x @ aligned
        gradient = basis.T @ ((x - corr * aligned) * derivative(mapped) / scale)
        return -corr + penalty, -gradient + 2.0 * penalty_grad_half

    fit = minimize(
        objective,
        start,
        jac=True,
        method="L-BFGS-B",
        bounds=[(-band, band)] * len(knots),
        options={"maxiter": cfg["maxiter"], "ftol": cfg["ftol"], "gtol": cfg["gtol"]},
    )
    if not fit.success or not np.isfinite(fit.fun):
        return z + global_shift, "solver_fallback", int(fit.nit)
    return z + basis @ fit.x, "fitted", int(fit.nit)


def rcorr_length_map(xf: np.ndarray, y: np.ndarray, band: float = 20.0):
    """Run length-scaled RCORR on padded GR curves."""
    n = len(xf) - 80
    x = xf[40 : 40 + n]
    u = np.arange(-30, n + 30.0)
    shift = correlation(x, y, u, band)
    step = 0.1524
    lm = (n - 1) * step
    cfg = {
        "smooth_window": 11,
        "smooth_order": 2,
        "knots": 9,
        "magnitude_penalty": 0.0001 * 36.576 / lm,
        "roughness_penalty": 0.1 * 36.576 / lm,
        "maxiter": 200,
        "ftol": 1e-12,
        "gtol": 1e-7,
    }
    m_m, status, _ = regularized_correlation(np.arange(n) * step, x, u * step, y, cfg, band * step, shift * step)
    return m_m / step, status
