import os
import sys
import unittest

import numpy as np


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from detection_policy import decide


class DetectionPolicyTests(unittest.TestCase):
    def yamnet_scores(self, frames, event_index=None, value=0.0):
        scores = np.zeros((frames, 521), dtype=np.float32)
        if event_index is not None:
            scores[:, event_index] = value
        return scores

    def test_other_class_peak_on_different_frame_does_not_cancel_scream(self):
        probs = np.array([
            [0.01, 0.01, 0.98],
            [0.01, 0.01, 0.98],
            [0.01, 0.01, 0.98],
            [0.90, 0.095, 0.005],
        ])
        self.assertEqual('scream', decide(probs)[0])

    def test_yamnet_screaming_support_relaxes_only_consecutive_count(self):
        probs = np.array([
            [0.02, 0.02, 0.96],
            [0.02, 0.02, 0.96],
            [0.10, 0.80, 0.10],
        ])
        self.assertEqual('normal', decide(probs)[0])
        scores = self.yamnet_scores(3, event_index=11, value=0.06)
        self.assertEqual('scream', decide(probs, yamnet_scores=scores)[0])

    def test_yamnet_alone_cannot_trigger_event(self):
        probs = np.array([[0.05, 0.90, 0.05]] * 3)
        scores = self.yamnet_scores(3, event_index=11, value=0.99)
        self.assertEqual('normal', decide(probs, yamnet_scores=scores)[0])

    def test_glass_requires_custom_and_yamnet_agreement(self):
        probs = np.array([[0.75, 0.20, 0.05]])
        self.assertEqual('normal', decide(probs)[0])
        scores = self.yamnet_scores(1, event_index=435, value=0.20)
        self.assertEqual('glass', decide(probs, yamnet_scores=scores)[0])


if __name__ == '__main__':
    unittest.main()
