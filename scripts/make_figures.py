"""Generate publication figures (PDF, SVG, and PNG previews) from saved result archives."""
import json
from pathlib import Path
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

REPO = Path(__file__).resolve().parents[1]
OUT_REG = REPO / 'results'
FIG_DIR = REPO / 'figures'
PNG_DIR = REPO / 'figures/previews'
FIG_DIR.mkdir(parents=True, exist_ok=True)
PNG_DIR.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
    'svg.fonttype': 'none',
    'pdf.fonttype': 42,
    'font.size': 7.5,
    'axes.titlesize': 8.0,
    'axes.labelsize': 7.5,
    'xtick.labelsize': 6.9,
    'ytick.labelsize': 6.9,
    'legend.fontsize': 6.6,
    'axes.spines.right': False,
    'axes.spines.top': False,
    'axes.linewidth': 0.75,
    'xtick.major.width': 0.7,
    'ytick.major.width': 0.7,
    'legend.frameon': False,
    'figure.facecolor': 'white',
    'savefig.facecolor': 'white',
})

P = {
    'blue_main': '#0F4D92',
    'blue_sec':  '#3775BA',
    'teal':      '#42949E',
    'green_pos': '#2E9E44',
    'green_bg':  '#DDF3DE',
    'aqua_bg':   '#E0F0F0',
    'lilac_bg':  '#EFEFF8',
    'peach_bg':  '#FBF5EA',
    'red_bg':    '#F6CFCB',
    'red_strong':'#B64342',
    'amber':     '#BA8C46',
    'indigo_mid':'#7884B4',
    'indigo_dark':'#484878',
    'dark':      '#272727',
    'mid':       '#606060',
    'light':     '#D8D8D8',
    'grid':      '#EAECEF',
}
W_DOUBLE = 180.0 / 25.4

tb = json.loads((OUT_REG / 'reference_table_bundle.json').read_text(encoding='utf-8'))
syn = json.loads((OUT_REG / 'reference_e2_synthetic_1032.json').read_text(encoding='utf-8'))
e3 = json.loads((OUT_REG / 'reference_e3_conditional_qa.json').read_text(encoding='utf-8'))
e4 = json.loads((OUT_REG / 'reference_e4_ablation_sensitivity.json').read_text(encoding='utf-8'))
kap = json.loads((OUT_REG / 'reference_kappa_sweep.json').read_text(encoding='utf-8'))
fc = np.load(OUT_REG / 'fig_cache.npz')


def panel_tag(ax, letter, x=-0.10, y=1.04):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=9.5, fontweight='bold',
            color=P['dark'], ha='left', va='bottom')


def save_pub(fig, name):
    fig.canvas.draw()
    fig.savefig(FIG_DIR / f'{name}.pdf', bbox_inches='tight', pad_inches=0.04)
    fig.savefig(FIG_DIR / f'{name}.svg', bbox_inches='tight', pad_inches=0.04)
    fig.savefig(PNG_DIR / f'{name}.png', dpi=260, bbox_inches='tight', pad_inches=0.04)
    plt.close(fig)


# ============================================================
# FIGURE 1: Two-Regime Motivation — Cohort A Failure Diagnosis (U1580A) vs Out-of-Sample Blind Win (U1445A)
# ============================================================
def make_fig1():
    fig, axes = plt.subplots(2, 2, figsize=(W_DOUBLE, 4.95), gridspec_kw={'hspace': 0.36, 'wspace': 0.25})

    # Top row: U1580A (Cohort A)
    z = fc['U1580A_z']
    gr_ref, gr_tar = fc['U1580A_gr_ref'], fc['U1580A_gr_tar_id']
    lcal = fc['U1580A_lcal']
    bad = (lcal <= 6.0) | (~np.isfinite(lcal)) | (z > 490.5)

    for ax in (axes[0, 0], axes[0, 1]):
        ax.grid(axis='y', color=P['grid'], lw=0.6, zorder=0)
        in_b, s0 = False, None
        for i in range(len(z)):
            if bool(bad[i]) and not in_b:
                s0 = z[i]; in_b = True
            elif not bool(bad[i]) and in_b:
                ax.axvspan(s0, z[i - 1], color=P['red_bg'], alpha=0.45, zorder=0)
                in_b = False
        if in_b:
            ax.axvspan(s0, z[-1], color=P['red_bg'], alpha=0.45, zorder=0)

    ax = axes[0, 0]
    ax.plot(z, gr_ref, color=P['dark'], lw=0.85, label='Reference HSGR')
    ax.plot(z, gr_tar, color=P['teal'], lw=0.80, alpha=0.82, label='Target HSGR')
    ax.set_ylim(-1.4, 3.8)
    ax.set_ylabel('Scaled HSGR (MAD)')
    ax.set_title(r'Cohort A U1580A: passes ($\mathrm{LCAL} \leq 6\,$in shaded)', loc='left', pad=4, fontsize=7.6)
    ax.legend(loc='upper left', ncol=2, fontsize=6.0, handlelength=1.4)
    panel_tag(ax, 'a', x=-0.13, y=1.04)

    ax = axes[0, 1]
    ax.plot(z, fc['U1580A_d_xc'], color=P['mid'], ls='--', lw=0.95, label='XCORR')
    ax.plot(z, fc['U1580A_d_v1'], color=P['red_strong'], lw=0.85, alpha=0.82, label='Base-RA')
    ax.plot(z, fc['U1580A_d_ras'], color=P['blue_main'], lw=1.35, label=r'NARA ($\lambda_{\mathrm{eff}}=0.86$)')
    ax.set_ylabel('Depth shift $d(z)$ (m)')
    ax.set_title('U1580A shifts (RHOM All 0.189 vs 0.197; Scr 0.096 vs 0.082)', loc='left', pad=4, fontsize=7.2)
    ax.legend(loc='lower left', ncol=3, fontsize=5.8, handlelength=1.3, columnspacing=0.9)
    panel_tag(ax, 'b', x=-0.13, y=1.04)

    # Bottom row: U1445A (Blind B valid-formation win across both RLA5 & RHOM)
    zu = fc['U1445A_z']
    ax = axes[1, 0]
    ax.grid(axis='y', color=P['grid'], lw=0.6, zorder=0)
    ax.plot(zu, fc['U1445A_d_xc'], color=P['mid'], ls='--', lw=0.95, label='XCORR')
    ax.plot(zu, fc['U1445A_d_v1'], color=P['red_strong'], lw=0.80, alpha=0.72, label='Base-RA')
    ax.plot(zu, fc['U1445A_d_rc'], color=P['amber'], lw=1.05, label='RCORR')
    ax.plot(zu, fc['U1445A_d_ras'], color=P['blue_main'], lw=1.35, label=r'NARA ($\lambda_{\mathrm{eff}}=1.82$)')
    ax.set_ylabel('Depth shift $d(z)$ (m)')
    ax.set_xlabel('Reference wireline depth WSF (m)')
    ax.set_title('Blind U1445A: NARA damps Base-RA shale jitter', loc='left', pad=4, fontsize=7.6)
    ax.legend(loc='lower right', ncol=2, fontsize=5.8, handlelength=1.3, columnspacing=0.9)
    panel_tag(ax, 'c', x=-0.13, y=1.04)

    ax = axes[1, 1]
    ax.grid(axis='y', color=P['grid'], lw=0.6, zorder=0)
    cs_v1, cs_rc, cs_na = fc['U1445A_cum_v1'], fc['U1445A_cum_rc'], fc['U1445A_cum_ras']
    ax.axhline(0.0, color=P['mid'], ls='--', lw=0.85)
    ax.fill_between(zu, cs_na, 0.0, where=(cs_na <= 0), color=P['green_bg'], alpha=0.6)
    ax.plot(zu, cs_v1, color=P['red_strong'], lw=1.0, label='Base-RA (0.0689)')
    ax.plot(zu, cs_rc, color=P['amber'], lw=1.1, label='RCORR (0.0692)')
    ax.plot(zu, cs_na, color=P['blue_main'], lw=1.4, label='NARA (0.0661; Scr 0.0660)')
    ax.set_ylabel(r'Cum. $\Delta\mathrm{SSE}$ ($\Omega\cdot\mathrm{m}$)$^2$')
    ax.set_xlabel('Reference wireline depth WSF (m)')
    ax.set_title('U1445A held-out RLA5 vs XCORR (also wins RHOM 0.0813)', loc='left', pad=4, fontsize=7.2)
    ax.legend(loc='lower left', fontsize=5.8, handlelength=1.4)
    panel_tag(ax, 'd', x=-0.13, y=1.04)

    save_pub(fig, 'figure_1')


# ============================================================
# FIGURE 2: Clean Modular Pipeline Schematic (180 x 74 mm)
# ============================================================
def make_fig2():
    W, H = W_DOUBLE, 74.0 / 25.4
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis('off')
    INK, MUTED, EDGE = '#243B4B', '#5C6F7E', '#CFD9E0'

    xs = [0.12, 1.88, 3.64, 5.40]
    bw, by, bh = 1.54, 1.38, 1.42
    cards = [
        ('01', 'MAD-Scaled Inputs &\n45-Response Bank', r'$\sigma \in \{0..4\}$ samples' + '\n' + r'$a \in \{0.8, 1.0, 1.2\}$' + '\n' + r'$b \in \{-0.2, 0, +0.2\}$', '#EFF4F9', P['indigo_mid']),
        ('02', 'Pass-1 Pilot DP\n(Huber, ' + r'$\delta = 1.0$)', r'Pilot $\lambda_0 = 0.05$' + '\nBounds spike weight\n' + r'Extracts residual $J_{\mathrm{data}}$', '#EDF4FB', P['blue_sec']),
        ('03', 'Residual-Adaptive\nStiffness Rule', r'$\lambda_{\mathrm{eff}} = 0.05 + 15 J_{\mathrm{data}}$' + '\n' + r'Synthetic: $\lambda \approx 0.14$' + '\nField: $0.53 - 9.51$', '#E8F3F1', P['teal']),
        ('04', 'Pass-2 Exact\nFwd-Bwd Graph DP', r'Optimal $(\eta^*, \phi^*)$' + '\n' + r'Cost profile $Q_k(v)$' + '\nFwd time: $2.7 - 21\\,$s', '#EBF3FA', P['blue_main']),
    ]
    for x0, (num, title, body, bg, accent) in zip(xs, cards):
        ax.add_patch(FancyBboxPatch((x0, by), bw, bh, boxstyle='round,pad=0.02,rounding_size=0.07',
                                    fc=bg, ec=EDGE, lw=0.9))
        ax.add_patch(Rectangle((x0 + 0.06, by + bh - 0.045), bw - 0.12, 0.042, fc=accent, ec='none'))
        ax.text(x0 + 0.10, by + bh - 0.17, num, fontsize=8.0, fontweight='bold', color=accent, va='center')
        ax.text(x0 + bw / 2, by + bh - 0.43, title, ha='center', va='center', fontsize=7.6, fontweight='bold', color=INK, linespacing=1.18)
        ax.text(x0 + bw / 2, by + 0.40, body, ha='center', va='center', fontsize=6.5, color=MUTED, linespacing=1.28)

    for i in range(3):
        x_a, x_b = xs[i] + bw + 0.04, xs[i + 1] - 0.04
        ym = by + bh / 2
        ax.add_patch(FancyArrowPatch((x_a, ym), (x_b, ym), arrowstyle='-|>', mutation_scale=9, lw=1.15, color='#6A7F8D'))
        if i == 1:
            ax.text((x_a + x_b) / 2, ym + 0.16, r'$J_{\mathrm{data}}$', ha='center', fontsize=7.0, fontweight='bold', color=P['blue_main'])
        elif i == 2:
            ax.text((x_a + x_b) / 2, ym + 0.16, r'$\lambda_{\mathrm{eff}}$', ha='center', fontsize=7.0, fontweight='bold', color=P['blue_main'])

    bxs = [0.12, 2.46, 4.96]
    bws = [2.18, 2.34, 1.98]
    b_cards = [
        ('a   Excess-Cost Profile $Q_k(v)$', '$0.005$-sublevel span separates narrow shifts\n(U1611A: $1.26\\,$m) from broad basins\n(U1580A: $5.07\\,$m); pre-gate DP check.', P['lilac_bg'], P['indigo_dark']),
        ('b   Two-Way Reciprocity Check', 'Empirical heuristic: run $y \\rightarrow x$ pass;\nrevert to XCORR if $a_{\\mathrm{fwd}} = a_{\\mathrm{rev}} \\in \\{0.8, 1.2\\}$\n(flags U1395B, U1437D).', P['peach_bg'], P['amber']),
        ('c   Held-Out Curve Evaluation', 'Freeze GR map; score RHOM,\n' + r'DTCO, RLA5 ($\Omega\cdot\mathrm{m}$ & $\log_{10}$), APLC' + '\non All and Screened masks.', '#F3F5F7', P['mid']),
    ]
    for x0, w0, (title, body, bg, accent) in zip(bxs, bws, b_cards):
        ax.add_patch(FancyBboxPatch((x0, 0.10), w0, 0.96, boxstyle='round,pad=0.02,rounding_size=0.07',
                                    fc=bg, ec=EDGE, lw=0.9))
        ax.add_patch(Rectangle((x0 + 0.06, 1.02), w0 - 0.12, 0.040, fc=accent, ec='none'))
        ax.text(x0 + w0 / 2, 0.87, title, ha='center', va='center', fontsize=7.3, fontweight='bold', color=INK)
        ax.text(x0 + w0 / 2, 0.42, body, ha='center', va='center', fontsize=6.35, color=MUTED, linespacing=1.24)

    for tx, rad in [(1.25, 0.06), (3.70, 0.02), (6.05, -0.02)]:
        ax.add_patch(FancyArrowPatch((6.18, by - 0.04), (tx, 1.11),
                                     connectionstyle=f'arc3,rad={rad}', arrowstyle='-|>', mutation_scale=8.5, lw=1.0, color='#6A7F8D'))

    save_pub(fig, 'figure_2')


# ============================================================
# FIGURE 3: Calibration Sweep (kappa) & 1,032-Case Synthetic Evaluation
# ============================================================
def make_fig3():
    fig, axes = plt.subplots(1, 3, figsize=(W_DOUBLE, 2.75), gridspec_kw={'width_ratios': [1.22, 1.25, 1.05]})

    ax = axes[0]
    kps = kap['kappas']
    syn_mae = [kap['spwla_01_06'][str(k)]['mae'] for k in kps]
    ax.axvspan(10, 20, color=P['aqua_bg'], alpha=0.55, zorder=0)
    ax.grid(axis='y', color=P['grid'], lw=0.6)
    lines = []
    specs = [('U1580A', 'U1580A (RHOM)', P['blue_main']), ('U1473A', 'U1473A (DTCO)', P['teal']),
             ('U1564F_primary', 'U1564F (RLA5)', P['amber']), ('U1395B_primary', 'U1395B (RLA5)', P['red_strong'])]
    for cid, lbl, col in specs:
        lines.append(ax.plot(kps, [kap['cohort_a'][cid][str(k)]['ratio_xc'] for k in kps], marker='o', ms=3.0, lw=1.15, color=col, label=lbl)[0])
    ax.axhline(1.0, color=P['mid'], ls='--', lw=0.8)
    ax.axvline(15.0, color=P['dark'], ls=':', lw=0.95)
    ax.text(15.4, 1.13, 'Locked\n' + r'$\kappa=15$', fontsize=6.2, color=P['dark'], fontweight='bold')
    ax.set_ylim(0.94, 1.48)
    ax2 = ax.twinx(); ax2.spines['top'].set_visible(False)
    lines.append(ax2.plot(kps, syn_mae, 's--', color=P['dark'], ms=2.6, lw=0.95, label='SPWLA 01-06')[0])
    ax2.set_ylim(0.18, 0.25); ax2.set_ylabel('SPWLA 01-06 MAE (samples)', fontsize=6.6)
    ax.set_xlabel(r'Residual slope $\kappa$'); ax.set_ylabel('Field RMSE / XCORR')
    ax.legend(lines, [l.get_label() for l in lines], loc='upper right', ncol=2, fontsize=5.15, handlelength=1.1, columnspacing=0.6)
    panel_tag(ax, 'a', x=-0.16, y=1.04)

    ax = axes[1]
    ax.grid(axis='y', color=P['grid'], lw=0.6, zorder=0)
    groups = [
        ('Prim.\n(216)', 1.5394, 0.2022, 0.1961),
        ('Ext.\n(108)', 2.1188, 0.4101, 0.3547),
        ('Box\n(54)', 0.5986, 0.1598, 0.1644),
        ('Tri+AR1\n(54)', 0.5972, 0.1629, 0.1504),
        ('VaryHet\n(54)', 0.4941, 0.1610, 0.1558),
        ('U1580A\n(48)', 0.6503, 0.2482, 0.2328),
    ]
    x = np.arange(len(groups)); w = 0.25
    ax.bar(x - w, [g[1] for g in groups], width=w, color=P['amber'], alpha=0.88, label='RCORR')
    ax.bar(x,     [g[2] for g in groups], width=w, color=P['red_strong'], alpha=0.85, label='Base-RA')
    ax.bar(x + w, [g[3] for g in groups], width=w, color=P['blue_main'], label='NARA')
    ax.set_xticks(x); ax.set_xticklabels([g[0] for g in groups], fontsize=5.25)
    ax.set_ylabel('Well-equal MAE (samples)'); ax.set_ylim(0, 2.30)
    ax.legend(loc='upper right', fontsize=6.0)
    panel_tag(ax, 'b', x=-0.12, y=1.04)

    ax = axes[2]
    ax.grid(axis='y', color=P['grid'], lw=0.6, zorder=0)
    bnames = ['identity', 'large', 'mild', 'shift035']
    xb = np.arange(len(bnames)); bw = 0.26
    rc_b = [0.1040, 1.0758, 0.1331, 0.1020]
    bra_b = [0.2607, 0.2118, 0.1490, 0.1382]
    nara_b = [syn['summary_b432'][k]['RA_SNR_HUB']['well_equal_mae'] for k in bnames]
    ax.bar(xb - bw, rc_b, width=bw, color=P['amber'], alpha=0.88, label='RCORR')
    ax.bar(xb,      bra_b, width=bw, color=P['red_strong'], alpha=0.85, label='Base-RA')
    ax.bar(xb + bw, nara_b, width=bw, color=P['blue_main'], label='NARA')
    ax.set_xticks(xb); ax.set_xticklabels(['Identity\nwarp', 'Large\nwarp', 'Mild\nwarp', 'Shift\n0.35'], fontsize=6.0)
    ax.set_ylabel('Well-equal MAE (samples)'); ax.set_ylim(0, 1.22)
    ax.text(2.35, 0.56, r'57.9% exact $\sigma=0$' + '\n(42.1% non-zero, $n=216$)', fontsize=5.5, color=P['blue_main'], fontweight='bold', ha='center')
    ax.legend(loc='upper right', fontsize=5.8)
    panel_tag(ax, 'c', x=-0.16, y=1.04)

    fig.tight_layout(w_pad=0.9)
    save_pub(fig, 'figure_3')


# ============================================================
# FIGURE 4: Hero 13-Well, 19-Endpoint Out-of-Sample Dumbbell & Summary
# ============================================================
def make_fig4():
    fig, axes = plt.subplots(1, 2, figsize=(W_DOUBLE, 4.95), gridspec_kw={'width_ratios': [1.55, 1.05]})

    ax = axes[0]
    y = np.arange(len(tb))[::-1]
    lbls = [f"{'A' if r['cohort']=='A_dev' else 'B'}: {r['case_id'].replace('_primary','')}" for r in tb]
    v1 = [r['all_rmse']['BASE_RA'] / r['all_rmse']['XCORR'] for r in tb]
    rc = [r['all_rmse']['RCORR_LENGTH'] / r['all_rmse']['XCORR'] for r in tb]
    ras = [r['all_rmse']['RA_SNR_HUB'] / r['all_rmse']['XCORR'] for r in tb]
    nrg = [r['all_rmse']['RA_SNR_HUB_NRGATE'] / r['all_rmse']['XCORR'] for r in tb]

    ax.axvspan(0.76, 1.0, color=P['green_bg'], alpha=0.28, zorder=0)
    ax.axvspan(1.05, 1.41, color=P['red_bg'], alpha=0.25, zorder=0)
    ax.axvline(1.0, color=P['dark'], ls='--', lw=0.95, label='XCORR (=1.00)')
    ax.axvline(1.05, color=P['red_strong'], ls=':', lw=0.75, alpha=0.7)
    ax.axhline(14.5, color=P['mid'], ls='-', lw=0.7)
    ax.text(1.40, 15.15, 'Cohort A (4 dev)', fontsize=6.3, color=P['mid'], ha='right', fontweight='bold')
    ax.text(1.40, 13.85, 'Cohort B (15 OOS)', fontsize=6.3, color=P['mid'], ha='right', fontweight='bold')

    for i in range(len(tb)):
        ax.plot([v1[i], nrg[i] if tb[i]['nrg'] else ras[i]], [y[i], y[i]], color='#D0D6DE', lw=1.0, zorder=1)
    ax.scatter(v1, y, color=P['red_strong'], marker='x', s=24, lw=1.05, label='Base-RA (4/19)', zorder=3)
    ax.scatter(rc, y - 0.16, color=P['amber'], marker='^', s=20, alpha=0.9, label='RCORR (12/19)', zorder=3)
    non_g_idx = [i for i, r in enumerate(tb) if not r['nrg']]
    g_idx = [i for i, r in enumerate(tb) if r['nrg']]
    ax.scatter([ras[i] for i in non_g_idx], [y[i] + 0.08 for i in non_g_idx], color=P['blue_main'], marker='o', s=24, label='NARA (11/19, DP)', zorder=4)
    ax.scatter([nrg[i] for i in g_idx], [y[i] for i in g_idx], color=P['green_pos'], marker='s', s=28, label='NARA (fallback, n=3)', zorder=5)

    ax.set_yticks(y); ax.set_yticklabels(lbls, fontsize=6.5)
    ax.set_xlim(0.76, 1.41)
    ax.set_xlabel('Held-out RMSE / XCORR RMSE (< 1.0 improves on XCORR)')
    # Place legend cleanly above axes so zero markers are occluded
    ax.legend(bbox_to_anchor=(0.0, 1.01, 1.0, 0.12), loc='lower left', ncol=3, fontsize=5.6,
              mode='expand', frameon=False, handletextpad=0.35, columnspacing=0.8)
    panel_tag(ax, 'a', x=-0.16, y=1.09)

    ax = axes[1]
    ax.grid(axis='y', color=P['grid'], lw=0.6, zorder=0)
    cats = ['DTW', 'Base-0', 'Base-RA', 'RCORR', 'NARA']
    keys = ['DTW_SLOPE', 'BASE_R0', 'BASE_RA', 'RCORR_LENGTH', 'RA_SNR_HUB_NRGATE']
    all_m = [np.mean([r['all_rmse'][k]/r['all_rmse']['XCORR'] for r in tb]) for k in keys]
    b_hm = [np.mean([r['healthy_rmse'][k]/r['healthy_rmse']['XCORR'] for r in tb if r['cohort']=='B_eval']) for k in keys]
    xx = np.arange(len(cats)); bw = 0.34
    b1 = ax.bar(xx - bw/2, all_m, width=bw, color=P['indigo_mid'], label='All 19 (full mask)')
    b2 = ax.bar(xx + bw/2, b_hm, width=bw, color=P['blue_main'], label='Blind B-15 (screened)')
    ax.axhline(1.0, color=P['dark'], ls='--', lw=0.85)
    for b in list(b1) + list(b2):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.004, f'{b.get_height():.3f}',
                ha='center', va='bottom', fontsize=5.9, rotation=90)
    ax.set_xticks(xx); ax.set_xticklabels(cats, fontsize=6.4, rotation=20, ha='right')
    ax.set_ylim(0.92, 1.145); ax.set_ylabel('Mean RMSE ratio vs XCORR')
    ax.legend(loc='upper right', fontsize=6.2)
    panel_tag(ax, 'b', x=-0.14, y=1.09)

    fig.tight_layout(w_pad=1.0)
    save_pub(fig, 'figure_4')


# ============================================================
# FIGURE 5: Blind Cohort B Regimes (U1611A win, U1467E smooth, U1437D rejected DP) — with RCORR on EVERY panel
# ============================================================
def make_fig5():
    fig, axes = plt.subplots(3, 2, figsize=(W_DOUBLE, 5.5), gridspec_kw={'width_ratios': [1.0, 1.0]})
    wells = [
        ('U1611A', '568', 'U1611A RLA5: 0.3410 -> 0.3189 (beats RCORR 0.3265)'),
        ('U1467E', '755', 'U1467E RLA5: smooth run (RCORR 0.1205 vs NARA 0.1265)'),
        ('U1437D', '1135', 'U1437D [rejected DP]: bilateral a=0.8 triggers XCORR'),
    ]
    for row, (w, nstr, note) in enumerate(wells):
        z = fc[f'{w}_z']
        ax = axes[row, 0]
        ax.grid(axis='y', color=P['grid'], lw=0.6)
        ax.plot(z, fc[f'{w}_d_xc'], color=P['mid'], ls='--', lw=0.95, label='XCORR')
        ax.plot(z, fc[f'{w}_d_v1'], color=P['red_strong'], lw=0.80, alpha=0.72, label='Base-RA')
        ax.plot(z, fc[f'{w}_d_rc'], color=P['amber'], lw=1.05, alpha=0.92, label='RCORR')
        ax.plot(z, fc[f'{w}_d_ras'], color=P['blue_main'], lw=1.35, label='NARA (DP)' if w == 'U1437D' else 'NARA')
        ax.set_ylabel('Shift $d(z)$ (m)')
        ax.set_title(f'{w} ($n={nstr}$): shifts', loc='left', fontsize=7.5, pad=3)
        panel_tag(ax, chr(97 + 2*row), x=-0.14, y=1.04)
        if row == 0:
            ax.legend(loc='lower left', ncol=2, fontsize=5.9)
        if row == 2:
            ax.set_xlabel('Reference wireline depth WSF (m)')

        axr = axes[row, 1]
        axr.grid(axis='y', color=P['grid'], lw=0.6)
        cv, crc, cn = fc[f'{w}_cum_v1'], fc[f'{w}_cum_rc'], fc[f'{w}_cum_ras']
        axr.axhline(0.0, color=P['mid'], ls='--', lw=0.85)
        axr.fill_between(z, cn, 0.0, where=(cn <= 0), color=P['green_bg'], alpha=0.6)
        axr.fill_between(z, cn, 0.0, where=(cn > 0), color=P['red_bg'], alpha=0.4)
        axr.plot(z, cv, color=P['red_strong'], lw=0.95, alpha=0.85, label='Base-RA - XCORR')
        axr.plot(z, crc, color=P['amber'], lw=1.10, label='RCORR - XCORR')
        axr.plot(z, cn, color=P['blue_main'], lw=1.35, label='NARA (rejected DP) - XCORR' if w == 'U1437D' else 'NARA - XCORR')
        axr.set_ylabel(r'Cum. $\Delta\mathrm{SSE}$ ($\Omega\cdot\mathrm{m}$)$^2$')
        axr.set_title(note, loc='left', fontsize=7.0, pad=3)
        panel_tag(axr, chr(98 + 2*row), x=-0.14, y=1.04)
        if row in (0, 2):
            axr.legend(loc='lower left' if row == 0 else 'upper left', fontsize=5.8)
        if row == 2:
            axr.set_xlabel('Reference wireline depth WSF (m)')

    fig.tight_layout(h_pad=0.9, w_pad=0.9)
    save_pub(fig, 'figure_5')


# ============================================================
# FIGURE 6: Exact Conditional Sub-Optimality Landscapes Q_k(v)
# ============================================================
def make_fig6():
    holes = ['U1580A', 'U1473A', 'U1445A', 'U1513E', 'U1611A', 'U1437D']
    fig, axes = plt.subplots(2, 3, figsize=(W_DOUBLE, 4.05), sharey=True)
    axes = axes.ravel()
    for i, w in enumerate(holes):
        fp = e3['field_profiles'][w]
        gm, dq = np.array(fp['grid_m']), np.array(fp['dq_mid'])
        mask = dq < 100
        ax = axes[i]
        ax.grid(axis='y', color=P['grid'], lw=0.6, zorder=0)
        ax.axhspan(0, 0.005, color=P['lilac_bg'], alpha=0.9, zorder=0)
        ax.plot(gm[mask], dq[mask], color=P['blue_main'], lw=1.35)
        ax.fill_between(gm[mask], dq[mask], 0.005, where=(dq[mask] <= 0.005), color=P['aqua_bg'], alpha=0.85)
        ax.axhline(0.005, color=P['mid'], ls=':', lw=0.85)
        ax.axvline(fp['mid_opt_m'], color=P['green_pos'], ls='--', lw=1.0)
        ax.set_ylim(-0.001, 0.038); ax.set_xlim(-3.15, 3.15)
        tag = ' [rejected DP]' if w == 'U1437D' else ''
        ax.set_title(f"{w}{tag} ($\lambda={fp['lam_eff']:.2f}$)", loc='left', fontsize=7.5, pad=3)
        wstr = f"$w_{{0.005}}={fp['mid_width_005_m']:.2f}\\,$m"
        if w == 'U1611A':
            wstr += ' (band-trunc.)'
        ax.text(0.97, 0.90, f"opt$={fp['mid_opt_m']:+.2f}\\,$m\n{wstr}",
                transform=ax.transAxes, ha='right', va='top', fontsize=5.9,
                bbox=dict(boxstyle='round,pad=0.2', fc='white', ec=P['light'], alpha=0.9))
        panel_tag(ax, chr(97 + i), x=-0.12, y=1.04)
        if i % 3 == 0:
            ax.set_ylabel(r'Excess objective $\Delta Q_{\mathrm{mid}}(v)$')
        if i >= 3:
            ax.set_xlabel(r'Midpoint offset $d_{\mathrm{mid}}$ (m)')
    fig.tight_layout(h_pad=0.9, w_pad=0.75)
    save_pub(fig, 'figure_6')


# ============================================================
# FIGURE 7: Component Ablation Pareto Trade-off & 4-Axis Sensitivity
# ============================================================
def make_fig7():
    S = e4['summary']
    fig = plt.figure(figsize=(W_DOUBLE, 5.25))
    gs_outer = fig.add_gridspec(2, 1, height_ratios=[1.05, 1.0], hspace=0.40)
    gs_top = gs_outer[0].subgridspec(1, 2, wspace=0.28)
    gs_bot = gs_outer[1].subgridspec(1, 3, wspace=0.92)

    # a: Pareto Frontier Trade-off Scatter (Primary 216 MAE vs Field All-19 Ratio)
    ax = fig.add_subplot(gs_top[0, 0])
    ax.grid(True, color=P['grid'], lw=0.6, zorder=0)
    ax.axhline(1.0, color=P['mid'], ls='--', lw=0.85)
    ax.axhspan(0.98, 1.0, color=P['green_bg'], alpha=0.32, zorder=0)
    pts = [
        ('ABL_Base-RA',         'Base-RA',           P['red_strong'], 'x',  (6,  -2)),
        ('ABL_RA-Fix0.05-Hub',  'w/o Adapt (0.05)',  P['red_strong'], 's',  (6,  -3)),
        ('ABL_RA-Fix2.0-Hub',   'w/o Adapt (2.0)',   P['amber'],      'D',  (-26, -12)),
        ('ABL_RA-SNR-L2',       r'w/o Huber ($L_2$)',P['indigo_mid'], '^',  (14,  15)),
        ('ABL_R0-SNR-Hub',      'w/o Broadening',    P['indigo_dark'],'v',  (0,    7)),
        ('ABL_RA-SNR-Hub',      'w/o Gate',          P['blue_sec'],   'o',  (12,  -7)),
        ('ABL_RA-SNR-Hub+Gate', 'NARA (full)',       P['green_pos'],  'P',  (8,   -3)),
    ]
    for key, lbl, col, mk, (dx, dy) in pts:
        xp, yp = S[key]['primary216_mae'], S[key]['all19_mean_ratio']
        ax.scatter([xp], [yp], color=col, marker=mk, s=42, zorder=4)
        ax.annotate(lbl, xy=(xp, yp), xytext=(dx, dy), textcoords='offset points',
                    fontsize=6.2, fontweight=('bold' if lbl == 'NARA (full)' else 'normal'), color=col)
    ax.plot([S['ABL_RA-Fix0.05-Hub']['primary216_mae'], S['ABL_RA-SNR-Hub']['primary216_mae'], S['ABL_RA-SNR-Hub+Gate']['primary216_mae']],
            [S['ABL_RA-Fix0.05-Hub']['all19_mean_ratio'], S['ABL_RA-SNR-Hub']['all19_mean_ratio'], S['ABL_RA-SNR-Hub+Gate']['all19_mean_ratio']],
            color=P['blue_sec'], ls=':', lw=1.0, zorder=2)
    ax.set_xlim(0.18, 0.40); ax.set_ylim(0.982, 1.098)
    ax.set_xlabel('Primary 216 MAE (samples, lower better)')
    ax.set_ylabel('Field All-19 RMSE / XCORR')
    ax.set_title('Synthetic vs field ablation trade-off', loc='left', pad=4)
    panel_tag(ax, 'a', x=-0.16, y=1.04)

    # b: Residual slope kappa plateau
    ax = fig.add_subplot(gs_top[0, 1])
    ks = [0, 5, 10, 15, 20, 30]
    ax.axvspan(10, 30, color=P['aqua_bg'], alpha=0.5, zorder=0)
    ax.grid(axis='y', color=P['grid'], lw=0.6)
    l1, = ax.plot(ks, [S[f'SEN_KAPPA_{k}']['all19_mean_ratio'] for k in ks], 'o-', color=P['blue_main'], lw=1.3, ms=3.8, label='Field All-19')
    l2, = ax.plot(ks, [S[f'SEN_KAPPA_{k}']['blind15_hth_mean_ratio'] for k in ks], 's-', color=P['green_pos'], lw=1.3, ms=3.5, label='Blind B-15 screened')
    ax.axhline(1.0, color=P['mid'], ls='--', lw=0.8)
    ax.axvline(15, color=P['dark'], ls=':', lw=0.95)
    ax.text(15.6, 1.028, 'Frozen\n' + r'$\kappa=15$', fontsize=6.4, color=P['dark'], fontweight='bold')
    ax.set_ylim(0.955, 1.078); ax.set_xlabel(r'Residual slope $\kappa$'); ax.set_ylabel('Mean RMSE / XCORR')
    ax2 = ax.twinx(); ax2.spines['top'].set_visible(False)
    l3, = ax2.plot(ks, [S[f'SEN_KAPPA_{k}']['primary216_mae'] for k in ks], '^--', color=P['red_strong'], lw=1.05, ms=3.5, label='Primary 216 MAE')
    ax2.set_ylim(0.18, 0.23); ax2.set_ylabel('Primary 216 MAE (samples)', fontsize=6.8)
    ax.legend([l1, l2, l3], [l.get_label() for l in (l1, l2, l3)], loc='upper right', fontsize=5.8)
    ax.set_title(r'Residual-slope $\kappa$ plateau (un-gated core)', loc='left', pad=4)
    panel_tag(ax, 'b', x=-0.16, y=1.04)

    # c: Huber threshold delta stability
    ax = fig.add_subplot(gs_bot[0, 0])
    ax.grid(axis='y', color=P['grid'], lw=0.6, zorder=0)
    dlbls = ['0.5', '1.0', '1.5', '2.0', r'$\infty$']
    dkeys = ['SEN_DELTA_0.5', 'SEN_DELTA_1.0', 'SEN_DELTA_1.5', 'SEN_DELTA_2.0', 'SEN_DELTA_inf']
    xd = np.arange(len(dlbls))
    la, = ax.plot(xd, [S[k]['all19_mean_ratio'] for k in dkeys], 'o-', color=P['blue_main'], lw=1.35, ms=3.8, label='All-19 mean', zorder=4)
    ax.axhline(1.0, color=P['mid'], ls='--', lw=0.8)
    ax.set_ylim(0.992, 1.012); ax.set_xticks(xd); ax.set_xticklabels(dlbls, fontsize=6.3)
    ax.set_xlabel(r'Huber $\delta$ (MAD)'); ax.set_ylabel('All-19 RMSE / XC')
    ax2 = ax.twinx(); ax2.spines['top'].set_visible(False)
    lb = ax2.bar(xd, [S[k]['all19_gt5'] for k in dkeys], width=0.38, color=P['red_bg'], edgecolor=P['red_strong'], lw=0.7, alpha=0.85, label='>5% count', zorder=1)
    ax2.set_ylim(0, 9); ax2.set_ylabel('>5% worse', color=P['red_strong'], fontsize=6.5)
    ax.legend([la, lb], ['All-19 mean', '>5% count'], loc='upper left', fontsize=5.5)
    ax.set_title(r'Huber $\delta$', loc='left', pad=4)
    panel_tag(ax, 'c', x=-0.24, y=1.04)

    # d: Pilot stiffness lam0 standalone axis
    ax = fig.add_subplot(gs_bot[0, 1])
    ax.grid(axis='y', color=P['grid'], lw=0.6, zorder=0)
    l0s = ['0.01', '0.05', '0.10', '0.20']
    x_l = np.arange(len(l0s))
    lp1, = ax.plot(x_l, [S[f'SEN_LAM0_{k}']['primary216_mae'] for k in l0s], 's-', color=P['green_pos'], lw=1.3, ms=3.6, label='Prim 216 MAE')
    lp2, = ax.plot(x_l, [S[f'SEN_LAM0_{k}']['box54_mae'] for k in l0s], 'o--', color=P['indigo_dark'], lw=1.1, ms=3.4, label='Box 54 MAE')
    ax.set_ylim(0.155, 0.225); ax.set_xticks(x_l); ax.set_xticklabels(l0s, fontsize=6.3)
    ax.set_xlabel(r'Pilot stiffness $\lambda_0$'); ax.set_ylabel('MAE (samples)')
    ax2 = ax.twinx(); ax2.spines['top'].set_visible(False)
    lp3, = ax2.plot(x_l, [S[f'SEN_LAM0_{k}']['blind15_hth_mean_ratio'] for k in l0s], 'd:', color=P['blue_main'], lw=1.1, ms=3.6, label='B-15 scr')
    ax2.set_ylim(0.975, 0.995); ax2.set_ylabel('B-15 RMSE/XC', fontsize=6.4, color=P['blue_main'])
    ax.legend([lp1, lp2, lp3], ['Prim 216', 'Box 54', 'B-15 scr'], loc='upper left', fontsize=5.4)
    ax.set_title(r'Pilot $\lambda_0$ stability', loc='left', pad=4)
    panel_tag(ax, 'd', x=-0.24, y=1.04)

    # e: Graph discretisation (h, dg) clean point-plot (no truncated bars or continuous line!)
    ax = fig.add_subplot(gs_bot[0, 2])
    ax.grid(axis='y', color=P['grid'], lw=0.6, zorder=0)
    discs = [('(10,0.25)', 'SEN_DISC_10_0.25'), ('(20,0.25)', 'SEN_DISC_20_0.25'), ('(40,0.25)', 'SEN_DISC_40_0.25'), ('(20,0.50)', 'SEN_DISC_20_0.50')]
    x_d = np.arange(len(discs))
    ld1 = ax.scatter(x_d - 0.1, [S[k]['primary216_mae'] for _, k in discs], color=P['indigo_dark'], marker='s', s=32, zorder=4, label='Prim 216')
    ax.set_ylim(0.18, 0.27); ax.set_xticks(x_d); ax.set_xticklabels(['10, 0.25', '20, 0.25*', '40, 0.25', '20, 0.50'], fontsize=5.8, rotation=15)
    ax.set_xlabel(r'Grid ($h$, $\Delta g$)'); ax.set_ylabel('Primary 216 MAE')
    ax2 = ax.twinx(); ax2.spines['top'].set_visible(False)
    ld2 = ax2.scatter(x_d + 0.1, [S[k]['all19_mean_ratio'] for _, k in discs], color=P['amber'], marker='D', s=30, zorder=4, label='All-19')
    ax2.axhline(1.0, color=P['mid'], ls='--', lw=0.75)
    ax2.set_ylim(0.990, 1.012); ax2.set_ylabel('All-19 RMSE/XC', fontsize=6.4, color=P['amber'])
    ax.legend([ld1, ld2], ['Prim 216', 'All-19'], loc='upper left', fontsize=5.4)
    ax.set_title('Discretisation knee', loc='left', pad=4)
    panel_tag(ax, 'e', x=-0.24, y=1.04)

    save_pub(fig, 'figure_7')


if __name__ == '__main__':
    make_fig1()
    make_fig2()
    make_fig3()
    make_fig4()
    make_fig5()
    make_fig6()
    make_fig7()
    print('All 7 figures exported (PDF + SVG + PNG).')
