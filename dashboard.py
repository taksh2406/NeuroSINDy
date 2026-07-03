import json
import http.server
import socketserver
import webbrowser
import threading
import time
import os
import sys
import numpy as np

# Add the repository root to python path so we can import sindy
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sindy.models import FitzHughNagumo
from sindy.differentiation import (
    finite_difference,
    savitzky_golay_difference,
    spline_difference,
    tv_difference
)
from sindy.core import FeatureLibrary, SINDyEngine
from run_experiments import get_true_coefficients

PORT = 8080

def run_sindy_pipeline(params):
    # Extract params with defaults
    noise_std = float(params.get('noise', 0.05))
    threshold = float(params.get('threshold', 0.05))
    diff_method = params.get('diff_method', 'tv')
    window_width = int(params.get('window_width', 10))
    auto_threshold = bool(params.get('auto_threshold', False))
    use_ensemble = bool(params.get('use_ensemble', False))
    
    # FHN parameters
    a = float(params.get('a', 0.7))
    b = float(params.get('b', 0.8))
    epsilon = float(params.get('epsilon', 0.08))
    
    # External current stimulus
    i_offset = float(params.get('i_offset', 0.5))
    i_amp = float(params.get('i_amp', 0.25))
    i_freq = float(params.get('i_freq', 0.1)) # Hz
    
    I_ext_func = lambda t: i_offset + i_amp * np.sin(2.0 * np.pi * i_freq * t)
    
    # Simulate
    fhn = FitzHughNagumo(a=a, b=b, epsilon=epsilon)
    t_eval = np.linspace(0.0, 40.0, 800)
    t_span = (t_eval[0], t_eval[-1])
    y0 = [-1.0, 0.0]
    
    sim_data = fhn.simulate(
        t_span=t_span,
        y0=y0,
        t_eval=t_eval,
        I_ext_func=I_ext_func,
        noise_std=noise_std,
        seed=42
    )
    
    X = np.column_stack([sim_data['v_meas'], sim_data['w_meas']])
    U = sim_data['I_ext']
    
    # Numerical derivatives (if not integral SINDy)
    if diff_method == 'integral':
        X_dot = None
    else:
        if diff_method == 'central':
            X_dot = finite_difference(t_eval, X, method='central')
        elif diff_method == 'savgol':
            X_dot = savitzky_golay_difference(t_eval, X, window_length=15, polyorder=3)
        elif diff_method == 'spline':
            s_val = 0 if noise_std == 0.0 else None
            X_dot = spline_difference(t_eval, X, s=s_val)
        elif diff_method == 'tv':
            X_dot = tv_difference(t_eval, X, alph=0.01, itern=15, scale='large')
            
    # SINDy Engine
    library = FeatureLibrary(degree=3, include_interaction_with_input=True)
    engine = SINDyEngine(threshold=threshold, alpha=1e-5, library=library)
    
    state_names = ['v', 'w']
    input_names = ['I_ext']
    
    if diff_method == 'integral':
        engine.fit_integral(t_eval, X, U, state_names=state_names, input_names=input_names, window_width=window_width)
    else:
        if auto_threshold:
            engine.select_threshold_bic(X, X_dot, U, state_names=state_names, input_names=input_names)
        if use_ensemble:
            engine.fit_ensemble(X, X_dot, U, state_names=state_names, input_names=input_names, n_models=30)
        else:
            engine.fit(X, X_dot, U, state_names=state_names, input_names=input_names)
            
    # Get equations
    eqs_text = engine.get_equations(precision=4, latex=False)
    eqs_latex = engine.get_equations(precision=4, latex=True)
    
    # Get true coefficients based on current feature names
    true_coefs = get_true_coefficients(fhn, engine.library.feature_names)
    
    # Simulate discovered
    discovered_trajectory = None
    try:
        discovered_trajectory = engine.simulate_discovered(y0=y0, t_eval=t_eval, I_ext_func=I_ext_func)
    except Exception as e:
        pass
        
    # Prepare JSON response
    response = {
        't': t_eval.tolist(),
        'v_true': sim_data['v_true'].tolist(),
        'w_true': sim_data['w_true'].tolist(),
        'v_meas': sim_data['v_meas'].tolist(),
        'w_meas': sim_data['w_meas'].tolist(),
        'I_ext': sim_data['I_ext'].tolist(),
        'eqs_text': eqs_text,
        'eqs_latex': eqs_latex,
        'feature_names': engine.library.feature_names,
        'discovered_coefficients': engine.coefficients.tolist(),
        'true_coefficients': true_coefs.tolist()
    }
    
    if X_dot is not None:
        response['dv_est'] = X_dot[:, 0].tolist()
        response['dw_est'] = X_dot[:, 1].tolist()
        response['dv_true'] = sim_data['dv_true'].tolist()
        response['dw_true'] = sim_data['dw_true'].tolist()
        
    if discovered_trajectory is not None:
        response['v_disc'] = discovered_trajectory[:, 0].tolist()
        response['w_disc'] = discovered_trajectory[:, 1].tolist()
        
    return response

# Dashboard HTML template with embedded JS and CSS
HTML_CONTENT = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NeuroSINDy Discovery Dashboard</title>
    
    <!-- Tailwind CSS for utility styling -->
    <script src="https://cdn.tailwindcss.com"></script>
    
    <!-- Chart.js for data visualization -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    
    <!-- MathJax for rendering LaTeX equation output -->
    <script>
        window.MathJax = {
            tex: {
                inlineMath: [['$', '$'], ['\\\\(', '\\\\)']]
            },
            svg: {
                fontCache: 'global'
            }
        };
    </script>
    <script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js" id="MathJax-script" async></script>
    
    <!-- Custom styling for dark glassmorphic look -->
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Fira+Code:wght@400;500&display=swap');
        
        body {
            font-family: 'Inter', sans-serif;
            background-color: #080b11;
            background-image: 
                radial-gradient(at 10% 20%, rgba(124, 58, 237, 0.08) 0px, transparent 50%),
                radial-gradient(at 90% 80%, rgba(6, 182, 212, 0.08) 0px, transparent 50%);
        }
        
        .glass-card {
            background: rgba(17, 24, 39, 0.7);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.05);
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        }
        
        .accent-gradient {
            background: linear-gradient(135deg, #7c3aed 0%, #06b6d4 100%);
        }
        
        /* Custom scrollbar */
        ::-webkit-scrollbar {
            width: 6px;
            height: 6px;
        }
        ::-webkit-scrollbar-track {
            background: #0d1117;
        }
        ::-webkit-scrollbar-thumb {
            background: #2f3542;
            border-radius: 3px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: #57606f;
        }
    </style>
</head>
<body class="text-slate-100 min-h-screen flex flex-col">

    <!-- Top Navigation Bar -->
    <header class="glass-card sticky top-0 z-40 border-b border-white/5 py-4 px-8 flex justify-between items-center">
        <div class="flex items-center gap-3">
            <div class="w-8 h-8 rounded-lg accent-gradient flex items-center justify-center font-bold text-white shadow-lg">N</div>
            <div>
                <h1 class="text-xl font-bold tracking-tight bg-gradient-to-r from-violet-400 to-cyan-400 bg-clip-text text-transparent">NeuroSINDy</h1>
                <p class="text-xs text-slate-400 font-medium">Dynamical Equation Discovery Engine</p>
            </div>
        </div>
        <div class="flex items-center gap-4">
            <span class="text-xs px-2.5 py-1 rounded-full bg-cyan-500/10 text-cyan-400 border border-cyan-500/20 font-medium">FitzHugh-Nagumo Model</span>
            <span class="text-xs px-2.5 py-1 rounded-full bg-violet-500/10 text-violet-400 border border-violet-500/20 font-medium flex items-center gap-1.5">
                <span class="w-1.5 h-1.5 rounded-full bg-green-500 animate-pulse"></span>
                Active Server
            </span>
        </div>
    </header>

    <div class="flex flex-1 overflow-hidden">
        
        <!-- Sidebar Controls -->
        <aside class="w-80 glass-card border-r border-white/5 p-6 overflow-y-auto flex flex-col gap-6 select-none">
            
            <!-- FHN Parameters -->
            <div>
                <h3 class="text-xs font-bold text-violet-400 uppercase tracking-wider mb-3">Model Parameters</h3>
                <div class="grid grid-cols-3 gap-2">
                    <div>
                        <label class="block text-[10px] text-slate-400 mb-1">a</label>
                        <input id="a" type="number" step="0.1" value="0.7" class="w-full bg-slate-900 border border-slate-700 text-slate-100 rounded px-2 py-1 text-sm focus:outline-none focus:border-violet-500">
                    </div>
                    <div>
                        <label class="block text-[10px] text-slate-400 mb-1">b</label>
                        <input id="b" type="number" step="0.1" value="0.8" class="w-full bg-slate-900 border border-slate-700 text-slate-100 rounded px-2 py-1 text-sm focus:outline-none focus:border-violet-500">
                    </div>
                    <div>
                        <label class="block text-[10px] text-slate-400 mb-1">&epsilon;</label>
                        <input id="epsilon" type="number" step="0.01" value="0.08" class="w-full bg-slate-900 border border-slate-700 text-slate-100 rounded px-2 py-1 text-sm focus:outline-none focus:border-violet-500">
                    </div>
                </div>
            </div>

            <!-- External Stimulus -->
            <div>
                <h3 class="text-xs font-bold text-violet-400 uppercase tracking-wider mb-3">Stimulus Current ($I_{\text{ext}}$)</h3>
                <div class="flex flex-col gap-3">
                    <div>
                        <div class="flex justify-between mb-1">
                            <label class="text-xs text-slate-400">Offset Current</label>
                            <span id="offset-val" class="text-xs font-mono text-cyan-400">0.50</span>
                        </div>
                        <input id="i_offset" type="range" min="0.0" max="1.5" step="0.05" value="0.50" oninput="document.getElementById('offset-val').innerText = parseFloat(this.value).toFixed(2)" class="w-full accent-cyan-500 bg-slate-900 h-1.5 rounded-lg appearance-none cursor-pointer">
                    </div>
                    <div>
                        <div class="flex justify-between mb-1">
                            <label class="text-xs text-slate-400">Sine Amplitude</label>
                            <span id="amp-val" class="text-xs font-mono text-cyan-400">0.25</span>
                        </div>
                        <input id="i_amp" type="range" min="0.0" max="1.0" step="0.05" value="0.25" oninput="document.getElementById('amp-val').innerText = parseFloat(this.value).toFixed(2)" class="w-full accent-cyan-500 bg-slate-900 h-1.5 rounded-lg appearance-none cursor-pointer">
                    </div>
                    <div>
                        <div class="flex justify-between mb-1">
                            <label class="text-xs text-slate-400">Sine Frequency (Hz)</label>
                            <span id="freq-val" class="text-xs font-mono text-cyan-400">0.10</span>
                        </div>
                        <input id="i_freq" type="range" min="0.01" max="0.5" step="0.01" value="0.10" oninput="document.getElementById('freq-val').innerText = parseFloat(this.value).toFixed(2)" class="w-full accent-cyan-500 bg-slate-900 h-1.5 rounded-lg appearance-none cursor-pointer">
                    </div>
                </div>
            </div>

            <!-- Noise & SINDy Settings -->
            <div>
                <h3 class="text-xs font-bold text-violet-400 uppercase tracking-wider mb-3">SINDy Engine Options</h3>
                <div class="flex flex-col gap-4">
                    <div>
                        <div class="flex justify-between mb-1">
                            <label class="text-xs text-slate-400">Measurement Noise (&sigma;)</label>
                            <span id="noise-val" class="text-xs font-mono text-cyan-400">0.05</span>
                        </div>
                        <input id="noise" type="range" min="0.0" max="0.1" step="0.01" value="0.05" oninput="document.getElementById('noise-val').innerText = parseFloat(this.value).toFixed(2)" class="w-full accent-cyan-500 bg-slate-900 h-1.5 rounded-lg appearance-none cursor-pointer">
                    </div>

                    <div>
                        <div class="flex justify-between mb-1">
                            <label class="text-xs text-slate-400">Sparsity Threshold (&lambda;)</label>
                            <span id="threshold-val" class="text-xs font-mono text-cyan-400">0.03</span>
                        </div>
                        <input id="threshold" type="range" min="0.005" max="0.2" step="0.005" value="0.03" oninput="document.getElementById('threshold-val').innerText = parseFloat(this.value).toFixed(3)" class="w-full accent-cyan-500 bg-slate-900 h-1.5 rounded-lg appearance-none cursor-pointer">
                    </div>

                    <div>
                        <label class="block text-xs text-slate-400 mb-1">Regression / Diff Method</label>
                        <select id="diff_method" onchange="toggleDiffMethod(this.value)" class="w-full bg-slate-900 border border-slate-700 text-slate-100 rounded px-2.5 py-1.5 text-sm focus:outline-none focus:border-violet-500">
                            <option value="integral" selected>Integral-Form SINDy (I-SINDy)</option>
                            <option value="tv">TVDiff Regularised</option>
                            <option value="savgol">Savitzky-Golay Diff</option>
                            <option value="spline">Cubic Smoothing Spline</option>
                            <option value="central">Central Finite Difference</option>
                        </select>
                    </div>

                    <div id="window_container" class="block">
                        <label class="block text-xs text-slate-400 mb-1">Integral Window (Steps)</label>
                        <input id="window_width" type="number" min="2" max="100" value="30" class="w-full bg-slate-900 border border-slate-700 text-slate-100 rounded px-2.5 py-1 text-sm focus:outline-none focus:border-violet-500">
                    </div>

                    <div id="opt_container" class="hidden">
                        <div class="flex items-center justify-between mb-1.5">
                            <label class="text-xs text-slate-300">Auto-tune threshold (BIC)</label>
                            <input id="auto_threshold" type="checkbox" class="w-4 h-4 rounded accent-violet-600 bg-slate-900 border-slate-700">
                        </div>
                        <div class="flex items-center justify-between">
                            <label class="text-xs text-slate-300">Ensemble bagging (E-SINDy)</label>
                            <input id="use_ensemble" type="checkbox" class="w-4 h-4 rounded accent-violet-600 bg-slate-900 border-slate-700">
                        </div>
                    </div>
                </div>
            </div>

            <!-- Run Button -->
            <button onclick="runSindy()" class="w-full accent-gradient hover:opacity-90 active:scale-[0.98] text-white font-semibold py-3 px-4 rounded-xl shadow-lg shadow-violet-500/10 transition flex items-center justify-center gap-2 mt-auto">
                <span id="btn-text">Discover Equations</span>
                <div id="loading-spinner" class="hidden w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin"></div>
            </button>
        </aside>

        <!-- Main Dashboard View -->
        <main class="flex-1 p-8 overflow-y-auto flex flex-col gap-6">
            
            <!-- Equations Comparison Cards -->
            <div class="grid grid-cols-2 gap-6">
                <!-- Ground Truth -->
                <div class="glass-card rounded-xl p-5 border-l-4 border-cyan-500/80">
                    <h4 class="text-xs font-bold text-cyan-400 uppercase tracking-wider mb-2">Ground Truth Equations</h4>
                    <div class="h-16 flex items-center text-lg font-mono">
                        $$\begin{aligned}
                        \frac{dv}{dt} &= v - \frac{1}{3}v^3 - w + I_{\text{ext}} \\
                        \frac{dw}{dt} &= \varepsilon(v + a - bw)
                        \end{aligned}$$
                    </div>
                </div>
                
                <!-- SINDy Discovered -->
                <div class="glass-card rounded-xl p-5 border-l-4 border-violet-500/80">
                    <h4 class="text-xs font-bold text-violet-400 uppercase tracking-wider mb-2">SINDy Discovered Equations</h4>
                    <div class="h-16 flex flex-col justify-center text-sm font-semibold text-slate-300 gap-1.5" id="eqs-latex-container">
                        <div id="eqs-latex-v" class="text-center font-mono">$$\text{Simulating}...$$</div>
                        <div id="eqs-latex-w" class="text-center font-mono"></div>
                    </div>
                </div>
            </div>

            <!-- Charts Section Grid -->
            <div class="grid grid-cols-2 gap-6">
                <!-- Trajectory Chart -->
                <div class="glass-card rounded-xl p-5 flex flex-col h-[400px]">
                    <h3 class="text-sm font-bold text-slate-200 mb-3 flex items-center gap-2">
                        <span class="w-2 h-2 rounded-full bg-violet-500"></span>
                        Trajectory Reconstruction comparison
                    </h3>
                    <div class="flex-1 relative">
                        <canvas id="trajChart"></canvas>
                    </div>
                </div>

                <!-- Phase Portrait Chart -->
                <div class="glass-card rounded-xl p-5 flex flex-col h-[400px]">
                    <h3 class="text-sm font-bold text-slate-200 mb-3 flex items-center gap-2">
                        <span class="w-2 h-2 rounded-full bg-cyan-500"></span>
                        Neural State Phase Portrait
                    </h3>
                    <div class="flex-1 relative">
                        <canvas id="phaseChart"></canvas>
                    </div>
                </div>
                
                <!-- Derivative Comparison Chart (Only for Derivative SINDy) -->
                <div id="deriv_chart_card" class="glass-card rounded-xl p-5 flex flex-col h-[400px]">
                    <h3 class="text-sm font-bold text-slate-200 mb-3 flex items-center gap-2">
                        <span class="w-2 h-2 rounded-full bg-pink-500"></span>
                        Derivative Estimation vs. Ground Truth
                    </h3>
                    <div class="flex-1 relative">
                        <canvas id="derivChart"></canvas>
                    </div>
                </div>

                <!-- Coefficient Comparison Heatmap/Bar Chart -->
                <div id="coeff_chart_card" class="glass-card rounded-xl p-5 flex flex-col h-[400px]">
                    <h3 class="text-sm font-bold text-slate-200 mb-3 flex items-center gap-2">
                        <span class="w-2 h-2 rounded-full bg-emerald-500"></span>
                        Discovered Library Coefficients
                    </h3>
                    <div class="flex-1 relative">
                        <canvas id="coeffChart"></canvas>
                    </div>
                </div>
            </div>

        </main>
    </div>

    <!-- JavaScript code handling client-side rendering and updates -->
    <script>
        let trajChart, phaseChart, derivChart, coeffChart;
        
        function toggleDiffMethod(val) {
            const windowContainer = document.getElementById('window_container');
            const optContainer = document.getElementById('opt_container');
            const derivCard = document.getElementById('deriv_chart_card');
            const coeffCard = document.getElementById('coeff_chart_card');
            
            if (val === 'integral') {
                windowContainer.classList.remove('hidden');
                windowContainer.classList.add('block');
                optContainer.classList.remove('block');
                optContainer.classList.add('hidden');
                derivCard.classList.add('hidden');
                coeffCard.classList.remove('col-span-1');
                coeffCard.classList.add('col-span-2');
            } else {
                windowContainer.classList.remove('block');
                windowContainer.classList.add('hidden');
                optContainer.classList.remove('hidden');
                optContainer.classList.add('block');
                derivCard.classList.remove('hidden');
                coeffCard.classList.remove('col-span-2');
                coeffCard.classList.add('col-span-1');
            }
            
            // Adjust canvas sizes when visibility changes
            setTimeout(() => {
                if (trajChart) trajChart.resize();
                if (phaseChart) phaseChart.resize();
                if (derivChart) derivChart.resize();
                if (coeffChart) coeffChart.resize();
            }, 100);
        }

        async function runSindy() {
            const btn = document.querySelector('button');
            const spinner = document.getElementById('loading-spinner');
            const btnText = document.getElementById('btn-text');
            
            btn.disabled = true;
            spinner.classList.remove('hidden');
            btnText.innerText = "Running Engine...";
            
            const params = {
                noise: parseFloat(document.getElementById('noise').value),
                threshold: parseFloat(document.getElementById('threshold').value),
                diff_method: document.getElementById('diff_method').value,
                window_width: parseInt(document.getElementById('window_width').value),
                auto_threshold: document.getElementById('auto_threshold').checked,
                use_ensemble: document.getElementById('use_ensemble').checked,
                a: parseFloat(document.getElementById('a').value),
                b: parseFloat(document.getElementById('b').value),
                epsilon: parseFloat(document.getElementById('epsilon').value),
                i_offset: parseFloat(document.getElementById('i_offset').value),
                i_amp: parseFloat(document.getElementById('i_amp').value),
                i_freq: parseFloat(document.getElementById('i_freq').value)
            };
            
            try {
                const response = await fetch('/api/run', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(params)
                });
                const data = await response.json();
                
                if (data.error) {
                    alert("Error: " + data.error);
                } else {
                    updateDashboard(data);
                }
            } catch(e) {
                alert("Failed to connect to API: " + e.message);
            } finally {
                btn.disabled = false;
                spinner.classList.add('hidden');
                btnText.innerText = "Discover Equations";
            }
        }
        
        function updateDashboard(data) {
            // Update Equations LaTeX
            document.getElementById('eqs-latex-v').innerHTML = "$$\\\\frac{dv}{dt} = " + data.eqs_latex[0] + "$$";
            document.getElementById('eqs-latex-w').innerHTML = "$$\\\\frac{dw}{dt} = " + data.eqs_latex[1] + "$$";
            MathJax.typesetPromise();
            
            // 1. Trajectory Chart
            const trajCtx = document.getElementById('trajChart').getContext('2d');
            if (trajChart) trajChart.destroy();
            
            const trajDatasets = [
                {
                    label: 'Measured v (Noisy)',
                    data: data.t.map((t, idx) => ({x: t, y: data.v_meas[idx]})),
                    borderColor: 'rgba(255, 255, 255, 0.15)',
                    borderWidth: 1,
                    pointRadius: 0,
                    showLine: true
                },
                {
                    label: 'True v',
                    data: data.t.map((t, idx) => ({x: t, y: data.v_true[idx]})),
                    borderColor: 'rgba(6, 182, 212, 0.8)',
                    borderWidth: 2,
                    pointRadius: 0,
                    showLine: true
                }
            ];
            
            if (data.v_disc) {
                trajDatasets.push({
                    label: 'Discovered v',
                    data: data.t.map((t, idx) => ({x: t, y: data.v_disc[idx]})),
                    borderColor: 'rgba(139, 92, 246, 0.9)',
                    borderWidth: 2,
                    borderDash: [5, 5],
                    pointRadius: 0,
                    showLine: true
                });
            }
            
            trajChart = new Chart(trajCtx, {
                type: 'scatter',
                data: { datasets: trajDatasets },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        x: { grid: { color: 'rgba(255, 255, 255, 0.05)' }, title: { display: true, text: 'Time (s)', color: '#94a3b8' }, ticks: { color: '#94a3b8' } },
                        y: { grid: { color: 'rgba(255, 255, 255, 0.05)' }, title: { display: true, text: 'Voltage / States', color: '#94a3b8' }, ticks: { color: '#94a3b8' } }
                    },
                    plugins: {
                        legend: { labels: { color: '#f1f5f9', font: { size: 11 } } }
                    }
                }
            });

            // 2. Phase Portrait Chart
            const phaseCtx = document.getElementById('phaseChart').getContext('2d');
            if (phaseChart) phaseChart.destroy();
            
            const phaseDatasets = [
                {
                    label: 'Measured Trajectory (v vs w)',
                    data: data.v_meas.map((v, idx) => ({x: v, y: data.w_meas[idx]})),
                    borderColor: 'rgba(255, 255, 255, 0.08)',
                    borderWidth: 1,
                    pointRadius: 0.5,
                    showLine: true
                },
                {
                    label: 'True Trajectory',
                    data: data.v_true.map((v, idx) => ({x: v, y: data.w_true[idx]})),
                    borderColor: 'rgba(6, 182, 212, 0.8)',
                    borderWidth: 2,
                    pointRadius: 0,
                    showLine: true
                }
            ];
            
            if (data.v_disc) {
                phaseDatasets.push({
                    label: 'Discovered Trajectory',
                    data: data.v_disc.map((v, idx) => ({x: v, y: data.w_disc[idx]})),
                    borderColor: 'rgba(139, 92, 246, 0.9)',
                    borderWidth: 2,
                    borderDash: [5, 5],
                    pointRadius: 0,
                    showLine: true
                });
            }
            
            phaseChart = new Chart(phaseCtx, {
                type: 'scatter',
                data: { datasets: phaseDatasets },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        x: { grid: { color: 'rgba(255, 255, 255, 0.05)' }, title: { display: true, text: 'v (Voltage)', color: '#94a3b8' }, ticks: { color: '#94a3b8' } },
                        y: { grid: { color: 'rgba(255, 255, 255, 0.05)' }, title: { display: true, text: 'w (Recovery Variable)', color: '#94a3b8' }, ticks: { color: '#94a3b8' } }
                    },
                    plugins: {
                        legend: { labels: { color: '#f1f5f9', font: { size: 11 } } }
                    }
                }
            });

            // 3. Derivative Chart (Only populated if derivative SINDy was run)
            if (data.dv_est) {
                const derivCtx = document.getElementById('derivChart').getContext('2d');
                if (derivChart) derivChart.destroy();
                
                derivChart = new Chart(derivCtx, {
                    type: 'scatter',
                    data: {
                        datasets: [
                            {
                                label: 'Estimated dv/dt',
                                data: data.t.map((t, idx) => ({x: t, y: data.dv_est[idx]})),
                                borderColor: 'rgba(236, 72, 153, 0.4)',
                                borderWidth: 1.5,
                                pointRadius: 0,
                                showLine: true
                            },
                            {
                                label: 'True dv/dt',
                                data: data.t.map((t, idx) => ({x: t, y: data.dv_true[idx]})),
                                borderColor: 'rgba(6, 182, 212, 0.8)',
                                borderWidth: 2,
                                pointRadius: 0,
                                showLine: true
                            }
                        ]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        scales: {
                            x: { grid: { color: 'rgba(255, 255, 255, 0.05)' }, title: { display: true, text: 'Time (s)', color: '#94a3b8' }, ticks: { color: '#94a3b8' } },
                            y: { grid: { color: 'rgba(255, 255, 255, 0.05)' }, title: { display: true, text: 'dv/dt', color: '#94a3b8' }, ticks: { color: '#94a3b8' } }
                        },
                        plugins: {
                            legend: { labels: { color: '#f1f5f9', font: { size: 11 } } }
                        }
                    }
                });
            }

            // 4. Coefficient Comparison Bar Chart
            const coeffCtx = document.getElementById('coeffChart').getContext('2d');
            if (coeffChart) coeffChart.destroy();
            
            // Build comparisons
            const labels = data.feature_names;
            const disc_v = data.discovered_coefficients.map(row => row[0]);
            const true_v = data.true_coefficients.map(row => row[0]);
            const disc_w = data.discovered_coefficients.map(row => row[1]);
            const true_w = data.true_coefficients.map(row => row[1]);
            
            coeffChart = new Chart(coeffCtx, {
                type: 'bar',
                data: {
                    labels: labels,
                    datasets: [
                        {
                            label: 'True v Coeffs',
                            data: true_v,
                            backgroundColor: 'rgba(6, 182, 212, 0.5)',
                            borderColor: 'rgba(6, 182, 212, 0.9)',
                            borderWidth: 1
                        },
                        {
                            label: 'Discovered v Coeffs',
                            data: disc_v,
                            backgroundColor: 'rgba(139, 92, 246, 0.6)',
                            borderColor: 'rgba(139, 92, 246, 1)',
                            borderWidth: 1
                        },
                        {
                            label: 'True w Coeffs',
                            data: true_w,
                            backgroundColor: 'rgba(245, 158, 11, 0.5)',
                            borderColor: 'rgba(245, 158, 11, 0.9)',
                            borderWidth: 1
                        },
                        {
                            label: 'Discovered w Coeffs',
                            data: disc_w,
                            backgroundColor: 'rgba(16, 185, 129, 0.6)',
                            borderColor: 'rgba(16, 185, 129, 1)',
                            borderWidth: 1
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        x: { grid: { color: 'rgba(255, 255, 255, 0.05)' }, ticks: { color: '#94a3b8' } },
                        y: { grid: { color: 'rgba(255, 255, 255, 0.05)' }, title: { display: true, text: 'Coefficient Value', color: '#94a3b8' }, ticks: { color: '#94a3b8' } }
                    },
                    plugins: {
                        legend: { labels: { color: '#f1f5f9', font: { size: 10 } } }
                    }
                }
            });
        }
        
        // Initialize dashboard setup on load
        window.addEventListener('load', () => {
            toggleDiffMethod('integral');
            runSindy();
        });
    </script>
</body>
</html>
"""

class DashboardHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Suppress logging to keep output clean
        pass

    def do_GET(self):
        if self.path == '/' or self.path == '/index.html':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(HTML_CONTENT.encode('utf-8'))
        else:
            self.send_error(404, 'File Not Found')
            
    def do_POST(self):
        if self.path == '/api/run':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            params = json.loads(post_data.decode('utf-8'))
            
            try:
                response_data = run_sindy_pipeline(params)
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(response_data).encode('utf-8'))
            except Exception as e:
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'error': str(e)}).encode('utf-8'))
        else:
            self.send_error(404, 'Not Found')

class DashboardHTTPServer(socketserver.TCPServer):
    allow_reuse_address = True

def main():
    handler = DashboardHandler
    print(f"Starting NeuroSINDy Local Server on port {PORT}...")
    
    server = DashboardHTTPServer(("", PORT), handler)
    
    # Auto-open browser in a background thread
    def open_browser():
        time.sleep(1.5)
        print(f"Auto-opening dashboard at http://localhost:{PORT}")
        webbrowser.open(f"http://localhost:{PORT}")
        
    thread = threading.Thread(target=open_browser)
    thread.daemon = True
    thread.start()
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping NeuroSINDy server...")
        server.shutdown()
        server.server_close()

if __name__ == '__main__':
    main()
