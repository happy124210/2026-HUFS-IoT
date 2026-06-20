import os
import sys
import unittest


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from model_selection import passes_deployment_gate, rank_key


def metrics(f1, false_alarm, glass, scream):
    return {
        'macro_f1': f1,
        'normal_false_alarm_rate': false_alarm,
        'recall': {'glass': glass, 'normal': 1.0 - false_alarm, 'scream': scream},
    }


class ModelSelectionTests(unittest.TestCase):
    def setUp(self):
        self.baseline = metrics(0.70, 0.04, 0.70, 0.70)

    def test_gate_accepts_validation_improvement(self):
        candidate = metrics(0.75, 0.05, 0.72, 0.71)
        self.assertTrue(passes_deployment_gate(candidate, self.baseline))

    def test_gate_rejects_false_alarm_regression(self):
        candidate = metrics(0.80, 0.06, 0.80, 0.80)
        self.assertFalse(passes_deployment_gate(candidate, self.baseline))

    def test_rank_key_prefers_lower_false_alarm_when_f1_ties(self):
        safer = metrics(0.80, 0.02, 0.75, 0.75)
        noisier = metrics(0.80, 0.04, 0.80, 0.80)
        self.assertGreater(rank_key(safer), rank_key(noisier))


if __name__ == '__main__':
    unittest.main()
