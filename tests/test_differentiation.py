import unittest
import numpy as np
from sindy.differentiation import (
    finite_difference,
    savitzky_golay_difference,
    spline_difference,
    tv_difference
)

class TestDifferentiation(unittest.TestCase):
    def setUp(self):
        self.t = np.linspace(0.0, 2 * np.pi, 200)
        self.dt = self.t[1] - self.t[0]
        self.X_clean = np.sin(self.t)
        self.dX_true = np.cos(self.t)
        
        # Add noise
        np.random.seed(42)
        self.noise = np.random.normal(0, 0.05, size=self.t.shape)
        self.X_noisy = self.X_clean + self.noise

    def test_finite_difference(self):
        # Clean data
        dX_fd = finite_difference(self.t, self.X_clean, method='central')
        self.assertEqual(dX_fd.shape, self.t.shape)
        
        # Ignore endpoints for comparison as boundary formulas are lower-order accuracy
        mae = np.mean(np.abs(dX_fd[5:-5] - self.dX_true[5:-5]))
        self.assertLess(mae, 0.01)

    def test_savitzky_golay_difference(self):
        # Clean data
        dX_sg = savitzky_golay_difference(self.t, self.X_clean, window_length=15, polyorder=3)
        self.assertEqual(dX_sg.shape, self.t.shape)
        
        mae = np.mean(np.abs(dX_sg[5:-5] - self.dX_true[5:-5]))
        self.assertLess(mae, 0.01)

        # Noisy data should smooth out noise compared to naive finite difference
        dX_fd_noisy = finite_difference(self.t, self.X_noisy, method='central')
        dX_sg_noisy = savitzky_golay_difference(self.t, self.X_noisy, window_length=21, polyorder=3)
        
        mae_fd = np.mean(np.abs(dX_fd_noisy - self.dX_true))
        mae_sg = np.mean(np.abs(dX_sg_noisy - self.dX_true))
        
        # Savitzky-Golay should do significantly better than finite differences on noisy data
        self.assertLess(mae_sg, mae_fd)

    def test_spline_difference(self):
        dX_spline = spline_difference(self.t, self.X_clean, s=0)
        self.assertEqual(dX_spline.shape, self.t.shape)
        
        mae = np.mean(np.abs(dX_spline[5:-5] - self.dX_true[5:-5]))
        self.assertLess(mae, 0.01)

    def test_tv_difference(self):
        # Test that TVDiff runs without errors and handles shapes correctly
        dX_tv = tv_difference(self.t, self.X_noisy, alph=0.01, itern=5, scale='large')
        self.assertEqual(dX_tv.shape, self.t.shape)
        
        # TVDiff on noisy data should yield better result than finite difference
        dX_fd_noisy = finite_difference(self.t, self.X_noisy, method='central')
        mae_fd = np.mean(np.abs(dX_fd_noisy[5:-5] - self.dX_true[5:-5]))
        mae_tv = np.mean(np.abs(dX_tv[5:-5] - self.dX_true[5:-5]))
        
        self.assertLess(mae_tv, mae_fd)

if __name__ == '__main__':
    unittest.main()
