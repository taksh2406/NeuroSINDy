import os
import argparse
import numpy as np
import pandas as pd
from sindy.models import FitzHughNagumo
from sindy.differentiation import (
    finite_difference,
    savitzky_golay_difference,
    spline_difference,
    tv_difference
)
from sindy.core import FeatureLibrary, SINDyEngine
from sindy.visualization import Visualizer

def get_true_coefficients(fhn, library_names):
    """
    Construct the true coefficient matrix for the FitzHugh-Nagumo model
    based on the library feature names.
    
    dv/dt = v - v^3/3 - w + I_ext
    dw/dt = epsilon*v + epsilon*a - epsilon*b*w
    """
    P = len(library_names)
    D = 2
    true_coefs = np.zeros((P, D))
    
    for i, name in enumerate(library_names):
        # dv/dt equations
        if name == 'v':
            true_coefs[i, 0] = 1.0
        elif name == 'v^3':
            true_coefs[i, 0] = -1.0 / 3.0
        elif name == 'w':
            true_coefs[i, 0] = -1.0
        elif name == 'I_ext':
            true_coefs[i, 0] = 1.0
            
        # dw/dt equations
        elif name == '1':
            true_coefs[i, 1] = fhn.epsilon * fhn.a
        elif name == 'v':
            true_coefs[i, 1] = fhn.epsilon
        elif name == 'w':
            true_coefs[i, 1] = -fhn.epsilon * fhn.b
            
    return true_coefs

def main():
    parser = argparse.ArgumentParser(
        description='NeuroSINDy: Discover FitzHugh-Nagumo equations from membrane voltage time series.'
    )
    parser.add_argument('--noise', type=float, default=0.0,
                        help='Standard deviation of Gaussian measurement noise (default: 0.0)')
    parser.add_argument('--threshold', type=float, default=0.05,
                        help='SINDy sparsity threshold (default: 0.05)')
    parser.add_argument('--diff-method', type=str, default='central',
                        choices=['central', 'savgol', 'spline', 'tv'],
                        help='Differentiation method (default: central)')
    parser.add_argument('--savgol-window', type=int, default=15,
                        help='Savitzky-Golay window size (default: 15)')
    parser.add_argument('--tv-alpha', type=float, default=0.01,
                        help='TVDiff regularization parameter alpha (default: 0.01)')
    parser.add_argument('--tv-iter', type=int, default=15,
                        help='TVDiff solver iterations (default: 15)')
    parser.add_argument('--use-ensemble', action='store_true',
                        help='Use Ensemble SINDy (E-SINDy) bootstrap bagging')
    parser.add_argument('--auto-threshold', action='store_true',
                        help='Use BIC (Bayesian Information Criterion) to select the optimal threshold')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for noise generation (default: 42)')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Directory to save results and plots (default: results)')
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 60)
    print(" NeuroSINDy: Dynamical Equation Discovery Engine ")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  Noise Level (std):  {args.noise}")
    print(f"  SINDy Threshold:    {args.threshold}")
    print(f"  Diff Method:        {args.diff_method.upper()}")
    if args.diff_method == 'savgol':
        print(f"  SavGol Window:      {args.savgol_window}")
    elif args.diff_method == 'tv':
        print(f"  TV alpha / iter:    {args.tv_alpha} / {args.tv_iter}")
    print(f"  Output Directory:   {args.output_dir}")
    print("-" * 60)
    
    # 1. Initialize FHN model (standard parameter regime for excitability / spiking)
    # a=0.7, b=0.8, epsilon=0.08
    a, b, epsilon = 0.7, 0.8, 0.08
    fhn = FitzHughNagumo(a=a, b=b, epsilon=epsilon)
    
    # Define external current function (dynamically rich time-varying current to ensure persistent excitation and break collinearity)
    I_ext_func = lambda t: 0.5 + 0.25 * np.sin(2.0 * np.pi * t / 10.0)
    
    # Time vector: 40 seconds simulated at dt = 0.05 (800 points)
    t_eval = np.linspace(0.0, 40.0, 800)
    t_span = (t_eval[0], t_eval[-1])
    y0 = [-1.0, 0.0]  # Start near resting potential
    
    print("Simulating FitzHugh-Nagumo model trajectories...")
    sim_data = fhn.simulate(
        t_span=t_span,
        y0=y0,
        t_eval=t_eval,
        I_ext_func=I_ext_func,
        noise_std=args.noise,
        seed=args.seed
    )
    
    # Stack measured states for SINDy input
    X = np.column_stack([sim_data['v_meas'], sim_data['w_meas']])
    U = sim_data['I_ext']
    
    # 2. Compute numerical derivatives
    print("Computing numerical derivatives...")
    if args.diff_method == 'central':
        X_dot = finite_difference(sim_data['t'], X, method='central')
    elif args.diff_method == 'savgol':
        X_dot = savitzky_golay_difference(sim_data['t'], X, window_length=args.savgol_window, polyorder=3)
    elif args.diff_method == 'spline':
        # Fit smoothing spline: if noise is 0, s=0, else s=None (auto-tuned)
        s_val = 0 if args.noise == 0.0 else None
        X_dot = spline_difference(sim_data['t'], X, s=s_val)
    elif args.diff_method == 'tv':
        print(f"  [TVDiff] Running total variation differentiation (this may take a few seconds)...")
        X_dot = tv_difference(sim_data['t'], X, alph=args.tv_alpha, itern=args.tv_iter, scale='large')
        
    # Calculate derivative error
    X_dot_true = np.column_stack([sim_data['dv_true'], sim_data['dw_true']])
    deriv_mae = np.mean(np.abs(X_dot - X_dot_true), axis=0)
    print(f"Derivative computation error (MAE):")
    print(f"  dv/dt error: {deriv_mae[0]:.6f}")
    print(f"  dw/dt error: {deriv_mae[1]:.6f}")
    print("-" * 60)
    
    # 3. Discover equations using SINDyEngine
    print("Running SINDy identification engine...")
    library = FeatureLibrary(degree=3, include_interaction_with_input=True)
    engine = SINDyEngine(threshold=args.threshold, alpha=1e-5, library=library)
    
    state_names = ['v', 'w']
    input_names = ['I_ext']
    
    if args.auto_threshold:
        print("Selecting optimal threshold using Bayesian Information Criterion (BIC)...")
        best_th = engine.select_threshold_bic(X, X_dot, U, state_names=state_names, input_names=input_names)
        print(f"  Optimal SINDy threshold found: {best_th:.5f}")
        
    if args.use_ensemble:
        print("Fitting Ensemble SINDy (E-SINDy) with bootstrap bagging...")
        engine.fit_ensemble(
            X, X_dot, U,
            state_names=state_names,
            input_names=input_names,
            n_models=50,
            subsample_ratio=0.8,
            inclusion_threshold=0.6
        )
    else:
        engine.fit(
            X, X_dot, U,
            state_names=state_names,
            input_names=input_names
        )
    
    # Get discovered equations
    discovered_eqs = engine.get_equations(precision=4, latex=False)
    print("\nDiscovered Equations:")
    for eq in discovered_eqs:
        print(f"  {eq}")
    print("-" * 60)
    
    # Print ground truth equations for comparison
    true_eq_v = f"dv/dt = v - 0.3333 v^3 - w + I_ext"
    true_eq_w = f"dw/dt = {epsilon*a:.4f} + {epsilon:.4f} v - {epsilon*b:.4f} w"
    print("Ground Truth Equations:")
    print(f"  {true_eq_v}")
    print(f"  {true_eq_w}")
    print("-" * 60)
    
    # 4. Simulate the discovered system of equations
    print("Simulating discovered dynamical equations...")
    try:
        discovered_trajectory = engine.simulate_discovered(
            y0=y0,
            t_eval=t_eval,
            I_ext_func=I_ext_func
        )
        traj_mae = np.mean(np.abs(discovered_trajectory - np.column_stack([sim_data['v_true'], sim_data['w_true']])), axis=0)
        print(f"Trajectory integration error (MAE against true clean trajectory):")
        print(f"  v trajectory error: {traj_mae[0]:.4f} mV")
        print(f"  w trajectory error: {traj_mae[1]:.4f}")
    except Exception as e:
        print(f"WARNING: Could not simulate discovered equations: {e}")
        discovered_trajectory = None
    print("-" * 60)
    
    # 5. Plotting results
    print("Generating publication-quality visualization figures...")
    viz = Visualizer()
    
    # Path settings
    traj_path = os.path.join(args.output_dir, 'trajectory_comparison.png')
    phase_path = os.path.join(args.output_dir, 'phase_portrait.png')
    coef_path = os.path.join(args.output_dir, 'coefficient_heatmap.png')
    pareto_path = os.path.join(args.output_dir, 'pareto_front.png')
    
    # Trajectory plot
    viz.plot_trajectories(
        t=t_eval,
        sim_data=sim_data,
        discovered_data=discovered_trajectory,
        save_path=traj_path
    )
    print(f"  Saved trajectory plot to: {traj_path}")
    
    # Phase Portrait plot
    viz.plot_phase_portrait(
        sim_data=sim_data,
        fhn_model=fhn,
        sindy_engine=engine,
        discovered_data=discovered_trajectory,
        save_path=phase_path
    )
    print(f"  Saved phase portrait plot to: {phase_path}")
    
    # Coefficient heatmap plot
    true_coefs = get_true_coefficients(fhn, engine.library.feature_names)
    viz.plot_coefficient_comparison(
        sindy_engine=engine,
        true_coefficients=true_coefs,
        save_path=coef_path
    )
    print(f"  Saved coefficient heatmap plot to: {coef_path}")
    
    # Pareto front plot
    viz.plot_pareto_front(
        X=X,
        X_dot=X_dot,
        U=U,
        state_names=state_names,
        input_names=input_names,
        library=library,
        save_path=pareto_path
    )
    print(f"  Saved Pareto front plot to: {pareto_path}")
    
    # Ensemble statistics plot
    if args.use_ensemble:
        ensemble_path = os.path.join(args.output_dir, 'ensemble_statistics.png')
        viz.plot_ensemble_statistics(
            sindy_engine=engine,
            save_path=ensemble_path
        )
        print(f"  Saved ensemble statistics plot to: {ensemble_path}")
        
    print("=" * 60)
    print("Discovery Process Completed Successfully!")
    print("=" * 60)

if __name__ == '__main__':
    main()
