#!/usr/bin/env python3
"""
generate_figures.py
===================
Generates publication-quality figures of FitzHugh-Nagumo neural dynamics
and SINDy equation discovery for the NeuroSINDy project.

Produces:
  results/fig1_neural_dynamics.png      - spike train, phase portrait, AP morphology
  results/fig2_bifurcation.png          - Hopf bifurcation analysis
  results/fig3_discovery_pipeline.png   - SINDy identification pipeline summary
"""

import os
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from scipy.signal import find_peaks
import seaborn as sns

warnings.filterwarnings('ignore')

from sindy.models import FitzHughNagumo
from sindy.differentiation import finite_difference, savitzky_golay_difference, tv_difference
from sindy.core import FeatureLibrary, SINDyEngine

# ─── Style ───────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor':   '#f8f9fa',
    'axes.grid':        True,
    'grid.alpha':       0.3,
    'grid.linestyle':   '--',
    'font.family':      'DejaVu Sans',
    'font.size':        11,
    'axes.titlesize':   13,
    'axes.titleweight': 'bold',
    'axes.labelsize':   11,
    'xtick.labelsize':  9,
    'ytick.labelsize':  9,
    'legend.fontsize':  9,
    'legend.framealpha': 0.9,
    'figure.dpi':       150,
    'savefig.dpi':      300,
    'savefig.bbox':     'tight',
})

C_V      = '#0d6efd'
C_W      = '#dc3545'
C_I      = '#fd7e14'
C_TRUE   = '#1a73e8'
C_DISC   = '#e8711a'
C_NOISY  = '#9e9e9e'
C_NULLV  = '#2e7d32'
C_NULLW  = '#7b1fa2'

OUTPUT = 'assets/figures'
os.makedirs(OUTPUT, exist_ok=True)

# ─── Simulate Data ───────────────────────────────────────────
a, b, eps = 0.7, 0.8, 0.08
fhn = FitzHughNagumo(a=a, b=b, epsilon=eps)
I_ext = lambda t: 0.5 + 0.25 * np.sin(2.0 * np.pi * t / 10.0)

t60   = np.linspace(0, 60, 1200)
t40   = np.linspace(0, 40, 800)

clean = fhn.simulate((0, 60), [-1.0, 0.0], t60, I_ext, noise_std=0.0, seed=0)
noisy = fhn.simulate((0, 40), [-1.0, 0.0], t40, I_ext, noise_std=0.05, seed=42)

v, w, I_arr = clean['v_true'], clean['w_true'], clean['I_ext']

print("Generating Figure 1: Neural Dynamics Overview...")


# ═══════════════════════════════════════════════════════════
# FIGURE 1 — Neural Dynamics Overview
# ═══════════════════════════════════════════════════════════
fig1 = plt.figure(figsize=(15, 11))
gs1  = gridspec.GridSpec(3, 2, figure=fig1, hspace=0.45, wspace=0.32,
                         height_ratios=[1.0, 1.6, 1.0])

ax_spike = fig1.add_subplot(gs1[0, :])
ax_phase = fig1.add_subplot(gs1[1, 0])
ax_zoom  = fig1.add_subplot(gs1[1, 1])
ax_recov = fig1.add_subplot(gs1[2, 0])
ax_curr  = fig1.add_subplot(gs1[2, 1])

# ── Panel 1: Spike train ──────────────────────────────────
ax_spike.fill_between(t60, -2.8, 2.8, where=(I_arr > 0.5),
                       color='#fff9c4', alpha=0.6, label='Strong stimulus')
ax_spike.fill_between(t60, -2.8, 2.8, where=(I_arr <= 0.5),
                       color='#e3f2fd', alpha=0.4, label='Weak stimulus')
ax_spike.plot(t60, v, color=C_V, lw=1.5, label='$v(t)$ — membrane voltage', zorder=3)
ax_spike.set_xlim(t60[0], t60[-1])
ax_spike.set_ylim(-2.8, 2.8)
ax_spike.set_xlabel('Time (s)')
ax_spike.set_ylabel('$v$ (mV)')
ax_spike.set_title('FitzHugh-Nagumo Action Potential Train under Periodic Stimulus')
ax_spike.legend(loc='upper right', ncol=3)

# Annotate a spike
peaks, _ = find_peaks(v, height=1.0, distance=50)
if len(peaks) >= 2:
    p = peaks[1]
    ax_spike.annotate('Action\nPotential', xy=(t60[p], v[p]),
                       xytext=(t60[p] + 1.5, v[p] + 0.4),
                       fontsize=8, color='black',
                       arrowprops=dict(arrowstyle='->', color='black', lw=1.2))

# ── Panel 2: Phase Portrait ───────────────────────────────
v_g = np.linspace(-2.5, 2.5, 22)
w_g = np.linspace(-0.3, 2.5, 22)
VG, WG = np.meshgrid(v_g, w_g)
I_mean = np.mean(I_arr)
DV = VG - VG**3 / 3 - WG + I_mean
DW = eps * (VG + a - b * WG)
speed = np.sqrt(DV**2 + DW**2)

ax_phase.streamplot(v_g, w_g, DV, DW,
                     color=speed, cmap='Greys',
                     density=1.3, linewidth=0.8, arrowsize=0.9)

# Nullclines
v_nc = np.linspace(-2.5, 2.5, 400)
w_vnull = v_nc - v_nc**3 / 3 + I_mean       # v-nullcline: dv/dt=0 → w = v - v^3/3 + I
w_wnull = np.linspace(-0.3, 2.5, 400)
v_wnull = b * w_wnull - a                    # w-nullcline: dw/dt=0 → v = bw - a

ax_phase.plot(v_nc, w_vnull, color=C_NULLV, lw=2.2, label='$v$-nullcline ($\dot{v}=0$)', zorder=3)
ax_phase.plot(v_wnull, w_wnull, color=C_NULLW, lw=2.2, label='$w$-nullcline ($\dot{w}=0$)', zorder=3)

# Gradient trajectory (color = time)
pts = np.array([v, w]).T.reshape(-1, 1, 2)
segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
lc = LineCollection(segs, cmap='plasma', norm=Normalize(t60[0], t60[-1]), lw=1.5, alpha=0.85)
lc.set_array(t60)
ax_phase.add_collection(lc)
cb = fig1.colorbar(lc, ax=ax_phase, pad=0.02, shrink=0.9)
cb.set_label('Time (s)', fontsize=8)

# Fixed point (approx intersection)
ax_phase.set_xlim(-2.6, 2.6)
ax_phase.set_ylim(-0.3, 2.6)
ax_phase.set_xlabel('Membrane potential $v$')
ax_phase.set_ylabel('Recovery variable $w$')
ax_phase.set_title('Phase Portrait — Nullclines & Limit Cycle\n(trajectory coloured by time)')
ax_phase.legend(loc='upper left', fontsize=8)

# ── Panel 3: Single AP Zoom ───────────────────────────────
if len(peaks) >= 2:
    p = peaks[1]
    sl = slice(max(0, p - 60), min(len(t60), p + 100))
    t_loc = t60[sl] - t60[max(0, p - 60)]
    v_loc = v[sl]

    ax_zoom.plot(t_loc, v_loc, color=C_V, lw=2.2)

    # Annotate key morphological features
    pk_rel = p - max(0, p - 60)
    ax_zoom.axhline(0, color='gray', ls='--', lw=0.9, alpha=0.7, label='$v = 0$ (rest ≈ 0)')
    ax_zoom.annotate('Peak', xy=(t_loc[pk_rel], v_loc[pk_rel]),
                      xytext=(t_loc[pk_rel] + 0.5, v_loc[pk_rel] + 0.25),
                      fontsize=8, arrowprops=dict(arrowstyle='->', lw=1.0))
    # Upstroke annotation
    up_idx = pk_rel - 10 if pk_rel > 10 else 0
    ax_zoom.annotate('', xy=(t_loc[pk_rel], v_loc[pk_rel]),
                      xytext=(t_loc[up_idx], v_loc[up_idx]),
                      arrowprops=dict(arrowstyle='->', color=C_V, lw=1.5))
    ax_zoom.text(t_loc[up_idx] - 0.3, (v_loc[pk_rel] + v_loc[up_idx]) / 2,
                  'Upstroke\n(depolarisation)', fontsize=7.5, color=C_V, ha='right')

    # AHP trough
    trough_idx = np.argmin(v_loc[pk_rel:pk_rel + 80]) + pk_rel
    ax_zoom.annotate('After-hyperpolarisation\n(AHP)', xy=(t_loc[trough_idx], v_loc[trough_idx]),
                      xytext=(t_loc[trough_idx] + 0.8, v_loc[trough_idx] - 0.3),
                      fontsize=7.5, arrowprops=dict(arrowstyle='->', lw=1.0))

    ax_zoom.set_xlabel('Time relative to spike (s)')
    ax_zoom.set_ylabel('$v$ (mV)')
    ax_zoom.set_title('Single Action Potential Morphology')
    ax_zoom.legend(fontsize=8)
else:
    ax_zoom.set_title('Single Action Potential (no spike detected)')

# ── Panel 4: Recovery variable ────────────────────────────
ax_recov.plot(t60, w, color=C_W, lw=1.5, label='$w(t)$ — recovery variable')
ax_recov.set_xlabel('Time (s)')
ax_recov.set_ylabel('$w$')
ax_recov.set_title('Recovery Variable Dynamics')
ax_recov.legend()

# ── Panel 5: External current ─────────────────────────────
ax_curr.plot(t60, I_arr, color=C_I, lw=1.5, label='$I_{\\mathrm{ext}}(t)$')
ax_curr.fill_between(t60, 0, I_arr, alpha=0.2, color=C_I)
ax_curr.set_xlabel('Time (s)')
ax_curr.set_ylabel('$I_{\\mathrm{ext}}$ (a.u.)')
ax_curr.set_title('External Stimulus Current')
ax_curr.legend()

fig1.suptitle('FitzHugh-Nagumo Neuron Model — Complete Dynamical Analysis',
               fontsize=15, fontweight='bold', y=1.01)

path1 = os.path.join(OUTPUT, 'fig1_neural_dynamics.png')
fig1.savefig(path1)
plt.close(fig1)
print(f"  Saved: {path1}")


# ═══════════════════════════════════════════════════════════
# FIGURE 2 — Bifurcation Analysis (Hopf Bifurcation)
# ═══════════════════════════════════════════════════════════
print("Generating Figure 2: Bifurcation Analysis...")

I_range = np.linspace(0.0, 1.4, 80)
v_max_arr, v_min_arr = [], []

for I_val in I_range:
    I_const = lambda t, _I=I_val: _I
    sim_bif = fhn.simulate((0, 40), [-1.0, 0.0],
                            np.linspace(0, 40, 600), I_const, noise_std=0.0)
    v_ss = sim_bif['v_true'][300:]   # discard first 50% transient
    v_max_arr.append(np.max(v_ss))
    v_min_arr.append(np.min(v_ss))

v_max_arr = np.array(v_max_arr)
v_min_arr = np.array(v_min_arr)

# Detect Hopf: where oscillation amplitude first exceeds a threshold
amp = v_max_arr - v_min_arr
hopf_idx = np.argmax(amp > 0.5)
I_hopf = I_range[hopf_idx]

fig2, axes2 = plt.subplots(1, 2, figsize=(13, 5.5))

# ── Left: Bifurcation diagram ─────────────────────────────
ax_bif = axes2[0]
ax_bif.plot(I_range, v_max_arr, color=C_TRUE, lw=2.0, label='$v$ max (limit cycle)')
ax_bif.plot(I_range, v_min_arr, color=C_DISC, lw=2.0, label='$v$ min (limit cycle)')
ax_bif.fill_between(I_range, v_min_arr, v_max_arr,
                     alpha=0.15, color=C_TRUE, label='Oscillation amplitude')
ax_bif.axvline(I_hopf, color='crimson', ls='--', lw=1.5,
                label=f'Hopf bifurcation ($I^*\\approx{I_hopf:.2f}$)')
ax_bif.set_xlabel('External current $I_{\\mathrm{ext}}$')
ax_bif.set_ylabel('Steady-state $v$ range')
ax_bif.set_title('Bifurcation Diagram\n$v$ amplitude vs. stimulus strength')
ax_bif.legend(fontsize=8)

# ── Right: Phase portraits at 3 I_ext values ─────────────
ax_port = axes2[1]
I_examples = [0.2, I_hopf, 0.9]
colors_ex  = ['#4caf50', 'crimson', '#1a73e8']
labels_ex  = [f'$I={{0.2}}$ (excitable)', f'$I\\approx{I_hopf:.2f}$ (Hopf)', '$I=0.9$ (limit cycle)']

for I_val, col, lbl in zip(I_examples, colors_ex, labels_ex):
    I_fn = lambda t, _I=I_val: _I
    sd = fhn.simulate((0, 50), [-1.0, 0.0], np.linspace(0, 50, 800), I_fn, noise_std=0.0)
    ax_port.plot(sd['v_true'][400:], sd['w_true'][400:], color=col, lw=1.5, label=lbl, alpha=0.85)

# Nullclines for reference (at I = 0.55)
v_nc2 = np.linspace(-2.5, 2.5, 300)
ax_port.plot(v_nc2, v_nc2 - v_nc2**3 / 3 + 0.55, '--', color='#aaa', lw=1.0, label='$v$-nullcline ($I=0.55$)')
w_nc2 = np.linspace(-0.5, 2.5, 300)
ax_port.plot(b * w_nc2 - a, w_nc2, ':', color='#aaa', lw=1.0, label='$w$-nullcline')

ax_port.set_xlim(-2.6, 2.6)
ax_port.set_ylim(-0.4, 2.5)
ax_port.set_xlabel('Membrane potential $v$')
ax_port.set_ylabel('Recovery variable $w$')
ax_port.set_title('Phase Portraits at Three Stimulus Levels\n(Excitable → Oscillatory transition)')
ax_port.legend(fontsize=8, loc='upper left')

fig2.suptitle('FitzHugh-Nagumo Hopf Bifurcation Analysis',
               fontsize=14, fontweight='bold', y=1.02)
path2 = os.path.join(OUTPUT, 'fig2_bifurcation.png')
fig2.savefig(path2)
plt.close(fig2)
print(f"  Saved: {path2}")


# ═══════════════════════════════════════════════════════════
# FIGURE 3 — SINDy Discovery Pipeline Summary
# ═══════════════════════════════════════════════════════════
print("Generating Figure 3: SINDy Discovery Pipeline...")

# Compute derivatives
X40     = np.column_stack([noisy['v_meas'], noisy['w_meas']])
X40_true = np.column_stack([noisy['dv_true'], noisy['dw_true']])
X_dot_fd  = finite_difference(t40, X40, method='central')
print("  Running TVDiff (this may take a moment)...")
X_dot_tv  = tv_difference(t40, X40, alph=0.2, itern=20, scale='large')

# Fit SINDy
library = FeatureLibrary(degree=3, include_interaction_with_input=True)
engine  = SINDyEngine(threshold=0.05, alpha=1e-5, library=library)
engine.fit(np.column_stack([noisy['v_true'], noisy['w_true']]),
           X40_true, noisy['I_ext'],
           state_names=['v', 'w'], input_names=['I_ext'])

# True coefficients
P = len(library.feature_names)
true_coefs = np.zeros((P, 2))
for i, name in enumerate(library.feature_names):
    if name == 'v':       true_coefs[i, 0] = 1.0
    elif name == 'v^3':   true_coefs[i, 0] = -1.0 / 3.0
    elif name == 'w':     true_coefs[i, 0] = -1.0;  true_coefs[i, 1] = -eps * b
    elif name == 'I_ext': true_coefs[i, 0] = 1.0
    elif name == '1':     true_coefs[i, 1] = eps * a
    elif name == 'v' and i > 0: true_coefs[i, 1] = eps

fig3 = plt.figure(figsize=(15, 10))
gs3  = gridspec.GridSpec(2, 3, figure=fig3, hspace=0.45, wspace=0.38)

ax_meas   = fig3.add_subplot(gs3[0, 0])
ax_deriv  = fig3.add_subplot(gs3[0, 1])
ax_theta  = fig3.add_subplot(gs3[0, 2])
ax_coef_v = fig3.add_subplot(gs3[1, 0:2])
ax_sim    = fig3.add_subplot(gs3[1, 2])

# ── Panel 1: Noisy measurements ──────────────────────────
t_show = t40[:200]
ax_meas.plot(t_show, noisy['v_true'][:200], color=C_V, lw=2.0, label='True $v(t)$', zorder=3)
ax_meas.plot(t_show, noisy['v_meas'][:200], '.', color=C_NOISY, ms=2.5,
              alpha=0.6, label='Noisy measurement ($\\sigma=0.05$)')
ax_meas.set_xlabel('Time (s)')
ax_meas.set_ylabel('$v$ (mV)')
ax_meas.set_title('Step 1: Noisy Measurements\n$v(t)$ with 5% Gaussian noise')
ax_meas.legend(fontsize=8)

# ── Panel 2: Derivative estimation ───────────────────────
ax_deriv.plot(t_show, X40_true[:200, 0], color=C_V, lw=2.0, label='True $\\dot{v}$', zorder=3)
ax_deriv.plot(t_show, X_dot_fd[:200, 0], color=C_NOISY, lw=1.0, alpha=0.7,
               label='Finite diff. (noisy)', zorder=1)
ax_deriv.plot(t_show, X_dot_tv[:200, 0], color=C_DISC, lw=1.8, ls='--',
               label='TVDiff (regularised)', zorder=2)
ax_deriv.set_xlabel('Time (s)')
ax_deriv.set_ylabel('$\\dot{v}$')
ax_deriv.set_title('Step 2: Derivative Estimation\nTVDiff vs. Finite Differences')
ax_deriv.legend(fontsize=8)

# ── Panel 3: Feature library heatmap ─────────────────────
# Show a subset of the library matrix (first 150 rows, normalised columns)
Theta = library.fit_transform(
    np.column_stack([noisy['v_true'], noisy['w_true']]),
    noisy['I_ext'],
    state_names=['v', 'w'], input_names=['I_ext']
)
Theta_show = Theta[:120:5, :]   # every 5th row for visual clarity
# Normalise each column for display
col_norm = np.max(np.abs(Theta_show), axis=0, keepdims=True) + 1e-10
Theta_normed = Theta_show / col_norm

im = ax_theta.imshow(Theta_normed.T, aspect='auto', cmap='RdBu_r',
                      vmin=-1, vmax=1, interpolation='nearest')
ax_theta.set_yticks(np.arange(len(library.feature_names)))
ax_theta.set_yticklabels(library.feature_names, fontsize=7)
ax_theta.set_xlabel('Data point index')
ax_theta.set_title('Step 3: Feature Library $\\boldsymbol{\\Theta}(\\mathbf{X},\\mathbf{U})$\n(column-normalised)')
plt.colorbar(im, ax=ax_theta, shrink=0.8, label='Normalised value')

# ── Panel 4: Coefficient comparison (bar chart) ──────────
feat_names = library.feature_names
x_idx = np.arange(len(feat_names))
width = 0.38

disc_v = engine.coefficients[:, 0]
true_v = true_coefs[:, 0]

ax_coef_v.bar(x_idx - width / 2, true_v, width, color=C_TRUE, alpha=0.85,
               label='Ground truth', edgecolor='k', linewidth=0.6)
ax_coef_v.bar(x_idx + width / 2, disc_v, width, color=C_DISC, alpha=0.85,
               label='Discovered (SINDy)', edgecolor='k', linewidth=0.6)
ax_coef_v.set_xticks(x_idx)
ax_coef_v.set_xticklabels(feat_names, rotation=40, ha='right', fontsize=8)
ax_coef_v.axhline(0, color='black', lw=0.8)
ax_coef_v.set_ylabel('Coefficient value')
ax_coef_v.set_title('Step 4: Discovered vs. True Coefficients\n$dv/dt$ equation (STLSQ sparse regression)')
ax_coef_v.legend(fontsize=9)

# ── Panel 5: Discovered vs true trajectory ───────────────
sim_disc = engine.simulate_discovered([-1.0, 0.0], t40, I_ext)
ax_sim.plot(t40, noisy['v_true'], color=C_V, lw=2.0, label='True $v(t)$')
ax_sim.plot(t40, sim_disc[:, 0], '--', color=C_DISC, lw=1.8, label='Discovered $v(t)$')
mae = np.mean(np.abs(sim_disc[:, 0] - noisy['v_true']))
ax_sim.set_xlabel('Time (s)')
ax_sim.set_ylabel('$v$ (mV)')
ax_sim.set_title(f'Step 5: Trajectory Validation\nMAE = {mae:.4f} mV')
ax_sim.legend(fontsize=8)

fig3.suptitle('SINDy Equation Discovery Pipeline — FitzHugh-Nagumo Model',
               fontsize=14, fontweight='bold', y=1.01)

path3 = os.path.join(OUTPUT, 'fig3_discovery_pipeline.png')
fig3.savefig(path3)
plt.close(fig3)
print(f"  Saved: {path3}")

print("\nAll figures generated successfully.")
print(f"  {path1}")
print(f"  {path2}")
print(f"  {path3}")
