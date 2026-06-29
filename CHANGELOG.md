# Changelog

All notable changes to NeuroSINDy are documented in this file.

## [1.0.0] — 2026-07-05

### Added
- `FitzHughNagumo` simulator with configurable parameters, noise, and external current
- `FeatureLibrary`: polynomial + input interaction feature matrix (up to degree 3)
- `SINDyEngine.fit`: Sequentially Thresholded Least Squares (STLSQ) with Ridge regularisation
- `SINDyEngine.predict`: derivative prediction from discovered coefficients
- `SINDyEngine.simulate_discovered`: ODE integration of discovered equations
- `SINDyEngine.get_equations`: symbolic equation string output (plain text and LaTeX)
- `SINDyEngine.select_threshold_bic`: BIC-based automatic sparsity threshold selection
- `SINDyEngine.fit_ensemble`: E-SINDy bootstrap bagging with inclusion probability quantification
- `finite_difference`: forward, backward, and central differencing
- `savitzky_golay_difference`: Savitzky-Golay polynomial smoothing differentiation
- `spline_difference`: cubic smoothing spline differentiation
- `tv_difference`: Total Variation Regularised Differentiation (TVDiff, both `small` and `large` scale modes)
- `Visualizer.plot_trajectories`: time-series overlay of true, noisy, and discovered trajectories
- `Visualizer.plot_phase_portrait`: phase plane with nullclines and vector field
- `Visualizer.plot_coefficient_comparison`: heatmap of discovered vs. true coefficients
- `Visualizer.plot_pareto_front`: accuracy–sparsity Pareto curve
- `Visualizer.plot_ensemble_statistics`: term inclusion probabilities and coefficient confidence bars
- Full CLI runner `run_experiments.py` with `--auto-threshold` and `--use-ensemble` flags
- 13 unit tests across all modules
