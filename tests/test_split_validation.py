import os
import sys
import unittest


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from split_validation import assert_disjoint_splits, origin_overlaps, source_unit, youtube_id


class SplitValidationTests(unittest.TestCase):
    def test_disjoint_splits_pass(self):
        assert_disjoint_splits({
            'train': ['glass:a'],
            'validation': ['normal:b'],
            'test': ['scream:c'],
        })

    def test_exact_group_overlap_fails(self):
        with self.assertRaisesRegex(ValueError, 'Data leakage'):
            assert_disjoint_splits({
                'train': ['scream:review_scream_pred_normal_001'],
                'validation': [],
                'test': ['scream:review_scream_pred_normal_001'],
            })

    def test_youtube_id_is_independent_of_category_prefix(self):
        first = 'normal:audioset_music_normal_mMm6VinyMiY_003000_004000'
        second = 'normal:audioset_home_normal_mMm6VinyMiY_003000_004000'
        self.assertEqual('mMm6VinyMiY', youtube_id(first))
        self.assertEqual('mMm6VinyMiY', youtube_id(second))
        self.assertEqual(source_unit(first), source_unit(second))

    def test_origin_overlap_detects_renamed_same_video(self):
        overlaps = origin_overlaps({
            'train': ['normal:audioset_music_normal_mMm6VinyMiY_003000_004000'],
            'validation': [],
            'test': ['normal:audioset_home_normal_mMm6VinyMiY_003000_004000'],
        })
        self.assertEqual({'mMm6VinyMiY': ['test', 'train']}, overlaps)


if __name__ == '__main__':
    unittest.main()
