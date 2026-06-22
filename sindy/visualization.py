import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class Visualizer:
    def __init__(self, style='seaborn-v0_8-whitegrid'):
        try:
            plt.style.use(style)
        except:
            plt.rcParams['axes.grid'] = True
            plt.rcParams['grid.alpha'] = 0.3
            
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.size'] = 11
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['axes.titlesize'] = 13
        plt.rcParams['legend.fontsize'] = 10
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10
        plt.rcParams['figure.dpi'] = 150
        
        self.colors = {'true': '#1a73e8','discovered': '#e8711a','meas': '#757575','v_color': '#0d6efd','w_color': '#dc3545','null_v': '#4caf50','null_w': '#9c27b0',}

    def plot_trajectories(self, t, sim_data, discovered_data=None, save_path=None, show=False):
        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        
        ax = axes[0]
        if not np.allclose(sim_data['v_true'], sim_data['v_meas']):
            ax.plot(t, sim_data['v_meas'], '.', color=self.colors['meas'], alpha=0.3, label='Measured v(t) (Noisy)')
            
        ax.plot(t, sim_data['v_true'], '-', color=self.colors['v_color'], linewidth=2.5, label='True v(t)')
        
        if discovered_data is not None:
            ax.plot(t, discovered_data[:, 0], '--', color=self.colors['discovered'], linewidth=2.0, label='Discovered v(t)')
            
        ax.set_ylabel('Voltage $v$ (mV)')
        ax.set_title('Membrane Voltage Trajectory')
        ax.legend(loc='upper right', frameon=True)
        
        ax = axes[1]
        if not np.allclose(sim_data['w_true'], sim_data['w_meas']):
            ax.plot(t, sim_data['w_meas'], '.', color=self.colors['meas'], alpha=0.3, label='Measured w(t) (Noisy)')
            
        ax.plot(t, sim_data['w_true'], '-', color=self.colors['w_color'], linewidth=2.5, label='True w(t)')
        
        if discovered_data is not None:
            ax.plot(t, discovered_data[:, 1], '--', color=self.colors['discovered'], linewidth=2.0, label='Discovered w(t)')
            
        ax.set_ylabel('Recovery $w$')
        ax.set_xlabel('Time $t$ (ms)')
        ax.set_title('Recovery Variable Trajectory')
        ax.legend(loc='upper right', frameon=True)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close(fig)

    def plot_phase_portrait(self, sim_data, fhn_model, sindy_engine=None, discovered_data=None, save_path=None, show=False):
        fig = plt.figure(figsize=(9, 8))
        
        v_min, v_max = np.min(sim_data['v_true']) - 0.5, np.max(sim_data['v_true']) + 0.5
        w_min, w_max = np.min(sim_data['w_true']) - 0.2, np.max(sim_data['w_true']) + 0.2
        
        v_grid = np.linspace(v_min, v_max, 20)
        w_grid = np.linspace(w_min, w_max, 20)
        V, W = np.meshgrid(v_grid, w_grid)
        
        mean_current = np.mean(sim_data['I_ext'])
        dv_field = V - (V**3)/3.0 - W + mean_current
        dw_field = fhn_model.epsilon * (V + fhn_model.a - fhn_model.b * W)
        
        speed = np.sqrt(dv_field**2 + dw_field**2)
        speed[speed == 0] = 1.0
        dv_norm = dv_field / speed
        dw_norm = dw_field / speed
        
        plt.quiver(V, W, dv_norm, dw_norm, speed, cmap='Blues', alpha=0.35, width=0.003, scale=30)
        
        v_arr = np.linspace(v_min, v_max, 200)
        null_v_true = v_arr - (v_arr**3)/3.0 + mean_current
        null_w_true = (v_arr + fhn_model.a) / fhn_model.b
        
        plt.plot(v_arr, null_v_true, '-', color=self.colors['null_v'], linewidth=2.0, label='True $v$-nullcline')
        plt.plot(v_arr, null_w_true, '-', color=self.colors['null_w'], linewidth=2.0, label='True $w$-nullcline')
        
        if sindy_engine is not None:
            grid_res = 100
            v_mesh, w_mesh = np.meshgrid(np.linspace(v_min, v_max, grid_res), np.linspace(w_min, w_max, grid_res))
            
            v_flat = v_mesh.ravel()
            w_flat = w_mesh.ravel()
            X_flat = np.column_stack([v_flat, w_flat])
            U_flat = np.ones(len(v_flat)) * mean_current
            
            derivatives = sindy_engine.predict(X_flat, U_flat)
            dv_pred = derivatives[:, 0].reshape(grid_res, grid_res)
            dw_pred = derivatives[:, 1].reshape(grid_res, grid_res)
            
            plt.contour(v_mesh, w_mesh, dv_pred, levels=[0.0], colors=self.colors['null_v'], linestyles='--', linewidths=1.8)
            plt.contour(v_mesh, w_mesh, dw_pred, levels=[0.0], colors=self.colors['null_w'], linestyles='--', linewidths=1.8)
            
            plt.plot([], [], '--', color=self.colors['null_v'], linewidth=1.8, label='Discovered $v$-nullcline')
            plt.plot([], [], '--', color=self.colors['null_w'], linewidth=1.8, label='Discovered $w$-nullcline')
                
        plt.plot(sim_data['v_true'], sim_data['w_true'], '-', color=self.colors['true'], linewidth=3.0, label='True Trajectory')
        
        if discovered_data is not None:
            plt.plot(discovered_data[:, 0], discovered_data[:, 1], '--', color=self.colors['discovered'], linewidth=2.5, label='Discovered Trajectory')
            
        plt.xlim(v_min, v_max)
        plt.ylim(w_min, w_max)
        plt.xlabel('Voltage $v$')
        plt.ylabel('Recovery $w$')
        plt.title(f'FitzHugh-Nagumo Phase Portrait (Mean $I_{{ext}}$ = {mean_current:.2f})')
        plt.legend(loc='lower right', frameon=True)
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close(fig)

    def plot_coefficient_comparison(self, sindy_engine, true_coefficients=None, save_path=None, show=False):
        coefs = sindy_engine.coefficients
        feat_names = sindy_engine.library.feature_names
        state_names = sindy_engine.state_names
        
        num_subplots = 2 if true_coefficients is not None else 1
        fig, axes = plt.subplots(1, num_subplots, figsize=(6 * num_subplots, 8), sharey=True)
        
        if num_subplots == 1:
            ax = axes
            sns.heatmap(coefs, annot=True, fmt=".3f", cmap="coolwarm", center=0.0,
                        yticklabels=feat_names, xticklabels=[f'd{s}/dt' for s in state_names],
                        ax=ax, cbar_kws={'label': 'Coefficient Value'})
            ax.set_title('Discovered Coefficients')
        else:
            ax = axes[0]
            sns.heatmap(true_coefficients, annot=True, fmt=".3f", cmap="coolwarm", center=0.0,
                        yticklabels=feat_names, xticklabels=[f'd{s}/dt' for s in state_names],
                        ax=ax, cbar=False)
            ax.set_title('True Coefficients')
            
            ax = axes[1]
            sns.heatmap(coefs, annot=True, fmt=".3f", cmap="coolwarm", center=0.0,
                        yticklabels=feat_names, xticklabels=[f'd{s}/dt' for s in state_names],
                        ax=ax, cbar_kws={'label': 'Coefficient Value'})
            ax.set_title('Discovered Coefficients')
            
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close(fig)

    def plot_pareto_front(self, X, X_dot, U=None, state_names=None, input_names=None, thresholds=None, library=None, save_path=None, show=False):
        from .core import SINDyEngine
        if thresholds is None:
            thresholds = np.logspace(-3, 0, 25)
            
        sparsity = []
        errors = []
        
        for th in thresholds:
            engine = SINDyEngine(threshold=th, library=library)
            engine.fit(X, X_dot, U, state_names=state_names, input_names=input_names)
            
            non_zeros = np.sum(engine.coefficients != 0)
            sparsity.append(non_zeros)
            
            X_dot_pred = engine.predict(X, U)
            mse = np.mean((X_dot - X_dot_pred)**2)
            errors.append(mse)
            
        fig = plt.figure(figsize=(8, 6))
        plt.plot(sparsity, errors, 'o-', color=self.colors['true'], linewidth=2.0, markersize=8)
        
        indices_to_label = [0, len(thresholds)//4, len(thresholds)//2, 3*len(thresholds)//4, len(thresholds)-1]
        for idx in indices_to_label:
            plt.annotate(
                f"$\lambda$={thresholds[idx]:.3f}",
                (sparsity[idx], errors[idx]),
                textcoords="offset points",
                xytext=(10, 5),
                ha='left',
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.2", fc="yellow", alpha=0.3)
            )
            
        plt.xlabel('Model Complexity (Number of Active Terms)')
        plt.ylabel('Reconstruction Error (Derivative MSE)')
        plt.title('SINDy Pareto Front: Sparsity vs Accuracy')
        plt.grid(True, which="both", ls="--", alpha=0.3)
        plt.yscale('log')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close(fig)

    def plot_ensemble_statistics(self, sindy_engine, save_path=None, show=False):
        """
        Plot ensemble inclusion probabilities and coefficient confidence intervals.
        """
        if not hasattr(sindy_engine, 'inclusion_probabilities'):
            raise ValueError("The provided SINDy engine has not been fitted using ensemble bagging.")
            
        inclusion_probs = sindy_engine.inclusion_probabilities
        coef_means = sindy_engine.coefficients_mean
        coef_stds = sindy_engine.coefficients_std
        feature_names = sindy_engine.library.feature_names
        state_names = sindy_engine.state_names
        
        D = len(state_names)
        fig, axes = plt.subplots(D, 2, figsize=(14, 4 * D), squeeze=False)
        
        x_indices = np.arange(len(feature_names))
        
        for j in range(D):
            # 1. Plot Inclusion Probabilities
            ax_prob = axes[j, 0]
            probs = inclusion_probs[:, j]
            bars = ax_prob.bar(x_indices, probs, color=self.colors['discovered'], alpha=0.8, edgecolor='k', linewidth=0.8)
            ax_prob.axhline(0.6, color='red', linestyle='--', linewidth=1.5, label='Inclusion Threshold (0.6)')
            
            ax_prob.set_xticks(x_indices)
            ax_prob.set_xticklabels(feature_names, rotation=45, ha='right', fontsize=9)
            ax_prob.set_ylabel('Inclusion Probability')
            ax_prob.set_ylim(0, 1.1)
            ax_prob.set_title(f'Ensemble Term Inclusion Probability for $d{state_names[j]}/dt$')
            ax_prob.grid(True, axis='y', ls='--', alpha=0.3)
            ax_prob.legend()
            
            # Highlight terms that are selected
            for idx, prob in enumerate(probs):
                if prob >= 0.6:
                    bars[idx].set_color(self.colors['true'])
            
            # 2. Plot Coefficient Mean & Std
            ax_coef = axes[j, 1]
            means = coef_means[:, j]
            stds = coef_stds[:, j]
            
            # Draw bar plot with error bars
            ax_coef.bar(x_indices, means, yerr=stds, color='#2c3e50', alpha=0.7, 
                        edgecolor='k', linewidth=0.8, capsize=4, error_kw={'ecolor': 'red', 'lw': 1.5})
            
            ax_coef.set_xticks(x_indices)
            ax_coef.set_xticklabels(feature_names, rotation=45, ha='right', fontsize=9)
            ax_coef.set_ylabel('Coefficient Value')
            ax_coef.set_title(f'Mean Discovered Coefficients with Std Dev for $d{state_names[j]}/dt$')
            ax_coef.grid(True, ls='--', alpha=0.3)
            
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close(fig)
