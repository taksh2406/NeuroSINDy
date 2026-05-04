import numpy as np
from scipy.integrate import solve_ivp

class FitzHughNagumo:
    def __init__(self, a=0.7, b=0.8, epsilon=0.08):
        self.a = a
        self.b = b
        self.epsilon = epsilon

    def f(self, t, y, I_ext_func):
        v, w = y
        I_ext = I_ext_func(t)
        dv = v - (v**3) / 3.0 - w + I_ext
        dw = self.epsilon * (v + self.a - self.b * w)
        return [dv, dw]

    def simulate(self, t_span, y0, t_eval, I_ext_func=None, noise_std=0.0, seed=None):
        if I_ext_func is None:
            I_ext_func = lambda t: 0.0

        sol = solve_ivp(
            fun=self.f,
            t_span=t_span,
            y0=y0,
            t_eval=t_eval,
            method='RK45',
            args=(I_ext_func,),
            rtol=1e-8,
            atol=1e-8
        )

        t = sol.t
        v = sol.y[0]
        w = sol.y[1]

        dv_true = np.zeros_like(v)
        dw_true = np.zeros_like(w)
        for i in range(len(t)):
            derivatives = self.f(t[i], [v[i], w[i]], I_ext_func)
            dv_true[i] = derivatives[0]
            dw_true[i] = derivatives[1]

        if noise_std > 0.0:
            if seed is not None:
                np.random.seed(seed)
            v_noise = np.random.normal(0, noise_std, size=v.shape)
            w_noise = np.random.normal(0, noise_std, size=w.shape)
            v_meas = v + v_noise
            w_meas = w + w_noise
        else:
            v_meas = v.copy()
            w_meas = w.copy()

        return {
            't': t,
            'v_true': v,
            'w_true': w,
            'v_meas': v_meas,
            'w_meas': w_meas,
            'dv_true': dv_true,
            'dw_true': dw_true,
            'I_ext': np.array([I_ext_func(ti) for ti in t])
        }
