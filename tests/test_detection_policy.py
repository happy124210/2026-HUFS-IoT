import json
import os
import sys
import unittest

import numpy as np


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from detection_policy import (
    DEPLOYMENT_POLICY,
    FINAL_FUSION_POLICY,
    LEGACY_CLASSIFIER_ONLY_POLICY,
    decide,
    decide_deployment,
    decide_final_fusion,
    decide_fusion,
    decide_legacy_classifier_only,
)


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

    def test_frame_aligned_scream_fusion_triggers(self):
        probs = np.array([
            [0.02, 0.10, 0.88],
            [0.10, 0.80, 0.10],
        ])
        scores = self.yamnet_scores(2)
        scores[0, 11] = 0.03
        self.assertEqual('scream', decide_fusion(probs, scores)[0])

    def test_support_on_different_frame_does_not_trigger(self):
        probs = np.array([
            [0.02, 0.10, 0.88],
            [0.10, 0.80, 0.10],
        ])
        scores = self.yamnet_scores(2)
        scores[1, 11] = 0.99
        self.assertEqual('normal', decide_fusion(probs, scores)[0])

    def test_yamnet_alone_cannot_trigger_event(self):
        probs = np.array([[0.05, 0.90, 0.05]] * 3)
        scores = self.yamnet_scores(3, event_index=11, value=0.99)
        self.assertEqual('normal', decide_fusion(probs, scores)[0])

    def test_glass_requires_custom_and_yamnet_agreement(self):
        probs = np.array([[0.75, 0.20, 0.05]])
        scores = self.yamnet_scores(1, event_index=435, value=0.20)
        self.assertEqual('glass', decide_fusion(probs, scores)[0])

    def test_deployment_policy_matches_validation_selected_fusion_policy(self):
        self.assertIs(DEPLOYMENT_POLICY, FINAL_FUSION_POLICY)
        self.assertEqual({
            'glass': {
                'strong_threshold': 0.95,
                'strong_min_frames': 2,
                'fusion_custom_threshold': 0.40,
                'fusion_yamnet_threshold': 0.10,
                'fusion_min_frames': 1,
            },
            'scream': {
                'strong_threshold': 0.55,
                'strong_min_frames': 3,
                'fusion_custom_threshold': 0.85,
                'fusion_yamnet_threshold': 0.02,
                'fusion_min_frames': 1,
            },
        }, FINAL_FUSION_POLICY)

    def test_code_policy_matches_saved_validation_selection(self):
        result_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), '..', 'test_results', 'training',
            'fusion_policy_selection_20260621.json',
        ))
        with open(result_path, encoding='utf-8') as handle:
            selected = json.load(handle)['selected_policy']
        self.assertEqual(FINAL_FUSION_POLICY, selected)

    def test_legacy_classifier_only_policy_remains_reproducible(self):
        self.assertEqual({'glass': 0.95, 'scream': 0.40}, LEGACY_CLASSIFIER_ONLY_POLICY['thresholds'])
        probs = np.array([[0.05, 0.45, 0.50]] * 3)
        self.assertEqual('scream', decide_legacy_classifier_only(probs)[0])

    def test_deployment_path_uses_yamnet_corroboration(self):
        probs = np.array([[0.02, 0.10, 0.88], [0.10, 0.80, 0.10]])
        no_support = self.yamnet_scores(2)
        support = self.yamnet_scores(2, event_index=11, value=0.03)
        self.assertEqual('normal', decide_deployment(probs, no_support)[0])
        self.assertEqual('scream', decide_deployment(probs, support)[0])
        self.assertEqual('scream', decide_final_fusion(probs, support)[0])


if __name__ == '__main__':
    unittest.main()
