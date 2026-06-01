import unittest
import numpy as np
from sindy.core import FeatureLibrary, SINDyEngine

class TestSINDyCore(unittest.TestCase):
    def test_library_creation_no_input(self):
        # 10 samples, 2 variables
        X = np.random.randn(10, 2)
        library = FeatureLibrary(degree=2)
        Theta = library.fit_transform(X, U=None, state_names=['x', 'y'])
        
        # Terms: 1, x, y, x^2, x*y, y^2
        # Total terms = 1 + 2 + 3 = 6
        self.assertEqual(Theta.shape, (10, 6))
        self.assertEqual(library.feature_names, ['1', 'x', 'y', 'x^2', 'x*y', 'y^2'])

    def test_library_creation_with_input(self):
        # 10 samples, 2 variables, 1 input
        X = np.random.randn(10, 2)
        U = np.random.randn(10, 1)
        library = FeatureLibrary(degree=2, include_interaction_with_input=True)
        Theta = library.fit_transform(X, U, state_names=['x', 'y'], input_names=['u'])
        
        # Polynomial terms: 1, x, y, x^2, x*y, y^2 (6 terms)
        # Input term: u (1 term)
        # Interaction terms: x*u, y*u (2 terms)
        # Total terms = 6 + 1 + 2 = 9
        self.assertEqual(Theta.shape, (10, 9))
        self.assertEqual(library.feature_names, ['1', 'x', 'y', 'x^2', 'x*y', 'y^2', 'u', 'x*u', 'y*u'])

    def test_sindy_exact_recovery(self):
        # Generate synthetic data for a simple system:
        # dx/dt = -0.5 * x - 1.0 * y
        # dy/dt = -1.0 * y
        # We simulate this analytically.
        t = np.linspace(0, 10, 1000)
        x = np.exp(-0.5 * t) + 2.0 * np.exp(-1.0 * t)
        y = np.exp(-1.0 * t)
        X = np.column_stack([x, y])
        
        # Analytical derivatives
        dx = -0.5 * np.exp(-0.5 * t) - 2.0 * np.exp(-1.0 * t)
        dy = -1.0 * np.exp(-1.0 * t)
        X_dot = np.column_stack([dx, dy])
        
        # Fit SINDy model
        engine = SINDyEngine(threshold=0.1, alpha=0.0)
        engine.fit(X, X_dot, state_names=['x', 'y'])
        
        # Feature names should include 'x' and 'y'
        # Coefficients matrix should be of shape (P, 2)
        # We verify that dx/dt = -0.5x - 1.0y and dy/dt = -1.0y
        # Find indices of x and y in library
        feat_names = engine.library.feature_names
        x_idx = feat_names.index('x')
        y_idx = feat_names.index('y')
        
        # Coefficients for dx/dt (column 0)
        self.assertAlmostEqual(engine.coefficients[x_idx, 0], -0.5, places=5)
        self.assertAlmostEqual(engine.coefficients[y_idx, 0], -1.0, places=5)
        
        # Other coefficients in column 0 should be 0
        non_zero_dx = np.where(engine.coefficients[:, 0] != 0)[0]
        self.assertEqual(len(non_zero_dx), 2)
        self.assertIn(x_idx, non_zero_dx)
        self.assertIn(y_idx, non_zero_dx)
        
        # Coefficients for dy/dt (column 1)
        self.assertAlmostEqual(engine.coefficients[y_idx, 1], -1.0, places=5)
        non_zero_dy = np.where(engine.coefficients[:, 1] != 0)[0]
        self.assertEqual(len(non_zero_dy), 1)
        self.assertIn(y_idx, non_zero_dy)
        
        # Test LaTeX output
        eqs_latex = engine.get_equations(latex=True)
        self.assertIn(r"\frac{dx}{dt}", eqs_latex[0])
        self.assertIn(r"\frac{dy}{dt}", eqs_latex[1])

if __name__ == '__main__':
    unittest.main()
