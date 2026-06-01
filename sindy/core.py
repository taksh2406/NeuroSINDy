import numpy as np
import itertools
from scipy.integrate import solve_ivp

class FeatureLibrary:
    def __init__(self, degree=3, include_interaction_with_input=True):
        self.degree = degree
        self.include_interaction_with_input = include_interaction_with_input
        self.feature_names = []

    def fit_transform(self, X, U=None, state_names=None, input_names=None):
        X = np.asarray(X)
        if X.ndim == 1:
            X = X[:, np.newaxis]
            
        M, D = X.shape
        
        if state_names is None:
            state_names = [f'x{i}' for i in range(D)]
        else:
            assert len(state_names) == D, f"Expected {D} state names, got {len(state_names)}"
            
        columns = [np.ones((M, 1))]
        names = ['1']
        
        for deg in range(1, self.degree + 1):
            for combos in itertools.combinations_with_replacement(range(D), deg):
                col = np.prod(X[:, combos], axis=1, keepdims=True)
                columns.append(col)
                name_parts = []
                for k, g in itertools.groupby(combos):
                    count = len(list(g))
                    var_name = state_names[k]
                    if count == 1:
                        name_parts.append(var_name)
                    else:
                        name_parts.append(f"{var_name}^{count}")
                names.append("*".join(name_parts))
                
        if U is not None:
            U = np.asarray(U)
            if U.ndim == 1:
                U = U[:, np.newaxis]
            
            M_u, K = U.shape
            assert M_u == M, f"Input U length ({M_u}) must match state X length ({M})."
            
            if input_names is None:
                input_names = [f'u{i}' for i in range(K)]
            else:
                assert len(input_names) == K, f"Expected {K} input names, got {len(input_names)}"
                
            for k in range(K):
                columns.append(U[:, k:k+1])
                names.append(input_names[k])
                
            if self.include_interaction_with_input:
                for i in range(D):
                    for k in range(K):
                        col = (X[:, i] * U[:, k])[:, np.newaxis]
                        columns.append(col)
                        names.append(f"{state_names[i]}*{input_names[k]}")
                        
        self.feature_names = names
        return np.hstack(columns)


class SINDyEngine:
    def __init__(self, threshold=0.05, max_iter=20, alpha=1e-5, library=None):
        self.threshold = threshold
        self.max_iter = max_iter
        self.alpha = alpha
        self.library = library if library is not None else FeatureLibrary(degree=3)
        self.coefficients = None
        self.state_names = None
        self.input_names = None
        
    def fit(self, X, X_dot, U=None, state_names=None, input_names=None):
        X = np.asarray(X)
        X_dot = np.asarray(X_dot)
        
        if X.ndim == 1:
            X = X[:, np.newaxis]
        if X_dot.ndim == 1:
            X_dot = X_dot[:, np.newaxis]
            
        M, D = X.shape
        
        self.state_names = state_names if state_names is not None else [f'x{i}' for i in range(D)]
        if U is not None:
            U_arr = np.asarray(U)
            if U_arr.ndim == 1:
                U_arr = U_arr[:, np.newaxis]
            K = U_arr.shape[1]
            self.input_names = input_names if input_names is not None else [f'u{i}' for i in range(K)]
        else:
            self.input_names = []
            
        Theta = self.library.fit_transform(X, U, state_names=self.state_names, input_names=self.input_names)
        P = Theta.shape[1]
        
        self.coefficients = np.zeros((P, D))
        
        for j in range(D):
            y = X_dot[:, j]
            
            if self.alpha > 0:
                lhs = Theta.T @ Theta + self.alpha * np.eye(P)
                rhs = Theta.T @ y
                xi = np.linalg.solve(lhs, rhs)
            else:
                xi = np.linalg.lstsq(Theta, y, rcond=None)[0]
                
            active = np.ones(P, dtype=bool)
            
            for _ in range(self.max_iter):
                small_indices = np.abs(xi) < self.threshold
                xi[small_indices] = 0
                
                new_active = ~small_indices
                
                if np.array_equal(new_active, active):
                    break
                active = new_active.copy()
                
                if not np.any(active):
                    break
                    
                Theta_active = Theta[:, active]
                if self.alpha > 0:
                    lhs = Theta_active.T @ Theta_active + self.alpha * np.eye(Theta_active.shape[1])
                    rhs = Theta_active.T @ y
                    xi[active] = np.linalg.solve(lhs, rhs)
                else:
                    xi[active] = np.linalg.lstsq(Theta_active, y, rcond=None)[0]
                    
            self.coefficients[:, j] = xi
            
        return self

    def predict(self, X, U=None):
        if self.coefficients is None:
            raise ValueError("Model must be fitted before making predictions.")
            
        Theta = self.library.fit_transform(X, U, state_names=self.state_names, input_names=self.input_names)
        return Theta @ self.coefficients

    def get_equations(self, precision=3, latex=False):
        if self.coefficients is None:
            raise ValueError("Model must be fitted before obtaining equations.")
            
        D = len(self.state_names)
        equations = []
        
        for j in range(D):
            state_var = self.state_names[j]
            lhs = f"d{state_var}/dt"
            if latex:
                lhs = r"\frac{d" + state_var + r"}{dt}"
                
            terms = []
            for i, coeff in enumerate(self.coefficients[:, j]):
                if np.abs(coeff) > 1e-12:
                    feat_name = self.library.feature_names[i]
                    
                    if latex:
                        feat_name = feat_name.replace("*", " ")
                        parts = feat_name.split()
                        for idx, p in enumerate(parts):
                            if "^" in p:
                                base, exp = p.split("^")
                                parts[idx] = f"{base}^{{{exp}}}"
                        feat_name = " ".join(parts)
                        
                    if feat_name == '1':
                        term_str = f"{coeff:.{precision}f}"
                    else:
                        term_str = f"{coeff:.{precision}f} {feat_name}"
                        
                    terms.append((coeff, term_str))
                    
            if not terms:
                rhs = "0"
            else:
                rhs = ""
                for idx, (coeff, term_str) in enumerate(terms):
                    if idx == 0:
                        rhs += term_str
                    else:
                        if coeff > 0:
                            rhs += f" + {term_str}"
                        else:
                            clean_term = term_str.lstrip("-")
                            rhs += f" - {clean_term}"
                            
            if latex:
                equations.append(f"${lhs} = {rhs}$")
            else:
                equations.append(f"{lhs} = {rhs}")
                
        return equations

    def simulate_discovered(self, y0, t_eval, I_ext_func=None):
        if self.coefficients is None:
            raise ValueError("Model must be fitted before simulation.")
            
        if I_ext_func is None:
            I_ext_func = lambda t: 0.0
            
        D = len(self.state_names)
        
        def discovered_rhs(t, y):
            X_row = y[np.newaxis, :]
            
            U_row = None
            if self.input_names:
                I_val = I_ext_func(t)
                U_row = np.array([[I_val]])
                
            Theta_row = self.library.fit_transform(X_row, U_row, state_names=self.state_names, input_names=self.input_names)
            dy = Theta_row @ self.coefficients
            return dy[0]
            
        t_span = (t_eval[0], t_eval[-1])
        sol = solve_ivp(
            fun=discovered_rhs,
            t_span=t_span,
            y0=y0,
            t_eval=t_eval,
            method='RK45',
            rtol=1e-8,
            atol=1e-8
        )
        return sol.y.T
