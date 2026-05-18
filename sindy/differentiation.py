import numpy as np
import scipy.signal
import scipy.interpolate
from scipy import sparse
from scipy.sparse import linalg as splin

def finite_difference(t, X, method='central'):
    t = np.asarray(t)
    X = np.asarray(X)
    
    is_1d = X.ndim == 1
    if is_1d:
        X = X[:, np.newaxis]
        
    M, D = X.shape
    X_dot = np.zeros_like(X)
    
    if M < 2:
        raise ValueError("Time series must have at least 2 points for finite difference.")
        
    dt = np.diff(t)
    
    if method == 'forward':
        for j in range(D):
            X_dot[:-1, j] = (X[1:, j] - X[:-1, j]) / dt
            X_dot[-1, j] = X_dot[-2, j]
    elif method == 'backward':
        for j in range(D):
            X_dot[1:, j] = (X[1:, j] - X[:-1, j]) / dt
            X_dot[0, j] = X_dot[1, j]
    elif method == 'central':
        for j in range(D):
            X_dot[1:-1, j] = (X[2:, j] - X[:-2, j]) / (t[2:] - t[:-2])
            X_dot[0, j] = (X[1, j] - X[0, j]) / dt[0]
            X_dot[-1, j] = (X[-1, j] - X[-2, j]) / dt[-1]
    else:
        raise ValueError(f"Unknown method: {method}. Must be 'central', 'forward', or 'backward'.")
        
    if is_1d:
        return X_dot.squeeze(-1)
    return X_dot

def savitzky_golay_difference(t, X, window_length=15, polyorder=3):
    t = np.asarray(t)
    X = np.asarray(X)
    
    dt_diff = np.diff(t)
    if not np.allclose(dt_diff, dt_diff[0], rtol=1e-3):
        dt = np.mean(dt_diff)
    else:
        dt = dt_diff[0]
        
    is_1d = X.ndim == 1
    if is_1d:
        X = X[:, np.newaxis]
        
    M, D = X.shape
    X_dot = np.zeros_like(X)
    
    if window_length >= M:
        window_length = M - 1 if (M - 1) % 2 == 1 else M - 2
        if window_length < polyorder + 1:
            window_length = polyorder + 1
            if window_length % 2 == 0:
                window_length += 1
                
    for j in range(D):
        X_dot[:, j] = scipy.signal.savgol_filter(
            X[:, j],
            window_length=window_length,
            polyorder=polyorder,
            deriv=1,
            delta=dt
        )
        
    if is_1d:
        return X_dot.squeeze(-1)
    return X_dot

def spline_difference(t, X, s=None):
    t = np.asarray(t)
    X = np.asarray(X)
    
    is_1d = X.ndim == 1
    if is_1d:
        X = X[:, np.newaxis]
        
    M, D = X.shape
    X_dot = np.zeros_like(X)
    
    for j in range(D):
        spline = scipy.interpolate.UnivariateSpline(t, X[:, j], s=s)
        X_dot[:, j] = spline.derivative()(t)
        
    if is_1d:
        return X_dot.squeeze(-1)
    return X_dot

def tv_difference(t, X, alph=0.1, itern=10, scale='large', ep=1e-6, precondflag=True, diffkernel='abs'):
    t = np.asarray(t)
    X = np.asarray(X)
    
    dt_diff = np.diff(t)
    dx = np.mean(dt_diff)
    
    is_1d = X.ndim == 1
    if is_1d:
        X = X[:, np.newaxis]
        
    M, D = X.shape
    X_dot = np.zeros_like(X)
    
    for j in range(D):
        data_col = X[:, j]
        u = _tv_reg_diff_core(data_col, itern=itern, alph=alph, scale=scale, ep=ep, dx=dx, precondflag=precondflag, diffkernel=diffkernel)
        
        if scale.lower() == 'small':
            u_grid = 0.5 * (u[:-1] + u[1:])
            X_dot[:, j] = u_grid
        else:
            X_dot[:, j] = u
            
    if is_1d:
        return X_dot.squeeze(-1)
    return X_dot

def _tv_reg_diff_core(data, itern, alph, u0=None, scale='large', ep=1e-6, dx=1.0, precondflag=True, diffkernel='abs', cgtol=1e-4, cgmaxit=100):
    n = len(data)
    
    if (scale.lower() == 'small'):
        d0 = -np.ones(n)/dx
        du = np.ones(n-1)/dx
        dl = np.zeros(n-1)
        dl[-1] = d0[-1]
        d0[-1] *= -1

        D = sparse.diags([dl, d0, du], [-1, 0, 1])
        DT = D.transpose()

        def A(x): 
            return (np.cumsum(x) - 0.5 * (x + x[0])) * dx

        def AT(x): 
            return np.concatenate([[sum(x[1:])/2.0], (sum(x)-np.cumsum(x)+0.5*x)[1:]])*dx

        if u0 is None:
            u0 = D * data

        u = u0.copy()
        ofst = data[0]
        ATb = AT(ofst - data)

        for ii in range(1, itern+1):
            if diffkernel == 'abs':
                Q = sparse.spdiags(1. / (np.sqrt((D * u)**2 + ep)), 0, n, n)
                L = dx * DT * Q * D
            elif diffkernel == 'sq':
                L = dx * DT * D
            else:
                raise ValueError('Invalid diffkernel value')

            g = AT(A(u)) + ATb + alph * L * u

            if precondflag:
                P = alph * sparse.spdiags(L.diagonal() + 1, 0, n, n)
            else:
                P = None

            def linop(v): 
                return (alph * L * v + AT(A(v)))
            
            linop_operator = splin.LinearOperator((n, n), linop)

            s, info_i = sparse.linalg.cg(linop_operator, g, x0=None, rtol=cgtol, maxiter=cgmaxit, callback=None, M=P)

            u = u - s

    elif (scale.lower() == 'large'):
        def A(v): 
            return np.cumsum(v)

        def AT(w): 
            return (sum(w) * np.ones(len(w)) - np.transpose(np.concatenate(([0.0], np.cumsum(w[:-1])))))
        
        c = np.ones(n)
        D = sparse.spdiags([-c, c], [0, 1], n, n) / dx
        mask = np.ones((n, n))
        mask[-1, -1] = 0.0
        D = sparse.dia_matrix(D.multiply(mask))
        DT = D.transpose()
        
        data_adjusted = data - data[0]
        if u0 is None:
            u0 = np.concatenate(([0], np.diff(data_adjusted)))
        u = u0.copy()
        ATd = AT(data_adjusted)

        for ii in range(1, itern + 1):
            if diffkernel == 'abs':
                Q = sparse.spdiags(1. / (np.sqrt((D * u)**2 + ep)), 0, n, n)
                L = DT * Q * D
            elif diffkernel == 'sq':
                L = DT * D
            else:
                raise ValueError('Invalid diffkernel value')

            g = AT(A(u)) - ATd + alph * L * u
            
            if precondflag:
                c_seq = np.cumsum(range(n, 0, -1))
                B = alph * L + sparse.spdiags(c_seq[::-1], 0, n, n)
                try:
                    R = sparse.dia_matrix(np.linalg.cholesky(B.todense()))
                    P = np.dot(R.transpose(), R)
                except np.linalg.LinAlgError:
                    P = alph * sparse.spdiags(L.diagonal() + 1, 0, n, n)
            else:
                P = None

            def linop(v): 
                return (alph * L * v + AT(A(v)))
            
            linop_operator = splin.LinearOperator((n, n), linop)

            s, info_i = sparse.linalg.cg(linop_operator, -g, x0=None, rtol=cgtol, maxiter=cgmaxit, callback=None, M=P)
            
            u = u + s

        u = u / dx

    return u
