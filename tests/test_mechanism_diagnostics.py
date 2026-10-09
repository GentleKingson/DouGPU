import unittest
import numpy as np
from scripts.mechanism_diagnostics import sample_consumption, noise_summary


class DiagnosticChecks(unittest.TestCase):
    def test_mixed_batch_counts_only_successful_steps(self):
        rows = [dict(event='start', session_id='a', updates=0, effective_batch=3),
                dict(event='train', session_id='a', updates=2, successful_steps=2, nonfinite_steps=1),
                dict(event='start', session_id='b', updates=2, effective_batch=6),
                dict(event='train', session_id='b', updates=3, successful_steps=1, nonfinite_steps=2)]
        self.assertEqual(sample_consumption(rows), 2*3+1*6)
        rows[-1]['updates'] = 4
        with self.assertRaises(ValueError):
            sample_consumption(rows)

    def test_gradient_variance(self):
        result = noise_summary([[1., 2.], [3., 4.]])
        self.assertEqual(result['variance_trace'], 4.)
        self.assertEqual(result['mean_gradient_squared_norm'], 13.)
        self.assertEqual(result['variance_over_mean_squared'], 4/13)
        self.assertEqual(noise_summary(np.ones((4, 3)))['variance_trace'], 0.)


if __name__ == '__main__':
    unittest.main()
