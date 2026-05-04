import unittest
import numpy as np
from sindy.models import FitzHughNagumo

class TestFitzHughNagumo(unittest.TestCase):
    def setUp(self):
        self.model = FitzHughNagumo(a=0.7, b=0.8, epsilon=0.08)

    def test_init(self):
        self.assertEqual(self.model.a, 0.7)
        self.assertEqual(self.model.b, 0.8)
        self.assertEqual(self.model.epsilon, 0.08)

    def test_f(self):
        # Test derivative calculations
        # y = [v, w] = [1.0, 0.5]
        # I_ext = 0.5
        I_ext_func = lambda t: 0.5
        dv, dw = self.model.f(0.0, [1.0, 0.5], I_ext_func)
        
        # dv/dt = v - v^3/3 - w + I_ext = 1.0 - 1.0/3.0 - 0.5 + 0.5 = 2/3 = 0.6666...
        # dw/dt = epsilon * (v + a - b * w) = 0.08 * (1.0 + 0.7 - 0.8 * 0.5) = 0.08 * (1.7 - 0.4) = 0.08 * 1.3 = 0.104
        self.assertAlmostEqual(dv, 2.0 / 3.0)
        self.assertAlmostEqual(dw, 0.104)

    def test_simulate_clean(self):
        t_span = (0.0, 10.0)
        t_eval = np.linspace(0.0, 10.0, 100)
        y0 = [0.0, 0.0]
        
        res = self.model.simulate(t_span, y0, t_eval)
        
        self.assertIn('t', res)
        self.assertIn('v_true', res)
        self.assertIn('w_true', res)
        self.assertIn('v_meas', res)
        self.assertIn('w_meas', res)
        self.assertIn('dv_true', res)
        self.assertIn('dw_true', res)
        self.assertIn('I_ext', res)
        
        self.assertEqual(len(res['t']), 100)
        self.assertTrue(np.array_equal(res['v_true'], res['v_meas']))
        self.assertTrue(np.array_equal(res['w_true'], res['w_meas']))

    def test_simulate_noisy(self):
        t_span = (0.0, 10.0)
        t_eval = np.linspace(0.0, 10.0, 100)
        y0 = [0.0, 0.0]
        noise_std = 0.1
        
        res = self.model.simulate(t_span, y0, t_eval, noise_std=noise_std, seed=42)
        
        self.assertFalse(np.array_equal(res['v_true'], res['v_meas']))
        self.assertFalse(np.array_equal(res['w_true'], res['w_meas']))
        
        # Verify noise standard deviation is roughly correct
        v_diff = res['v_meas'] - res['v_true']
        self.assertAlmostEqual(np.std(v_diff), noise_std, places=1)

if __name__ == '__main__':
    unittest.main()
