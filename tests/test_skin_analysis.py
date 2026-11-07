import unittest

import cv2
import numpy as np

from backend.image_processor import ImageProcessor
from backend.skin_analysis import SkinAnalyzer


class SkinAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.analyzer = SkinAnalyzer()
        self.image = np.full((180, 180, 3), (100, 145, 195), dtype=np.uint8)
        self.skin_mask = np.full((180, 180), 255, dtype=np.uint8)

    def test_detects_a_localized_red_pimple_candidate(self):
        cv2.circle(self.image, (60, 60), 6, (65, 70, 220), -1)

        count, _, regions = self.analyzer.detect_acne_like(self.image, self.skin_mask)

        self.assertGreaterEqual(count, 1)
        self.assertEqual(count, len(regions))

    def test_does_not_detect_bumps_in_uniform_skin(self):
        count, _, regions = self.analyzer.detect_acne_like(self.image, self.skin_mask)

        self.assertEqual(count, 0)
        self.assertEqual(regions, [])

    def test_detects_a_localized_dark_pimple_candidate(self):
        cv2.circle(self.image, (120, 100), 6, (45, 75, 115), -1)

        count, _, regions = self.analyzer.detect_acne_like(self.image, self.skin_mask)

        self.assertGreaterEqual(count, 1)
        self.assertEqual(count, len(regions))

    def test_rejects_a_thin_dark_hair_as_a_pimple(self):
        cv2.line(self.image, (30, 90), (150, 100), (35, 55, 75), 2)

        count, _, regions = self.analyzer.detect_acne_like(self.image, self.skin_mask)

        self.assertEqual(count, 0)
        self.assertEqual(regions, [])

    def test_rejects_a_thin_dark_hair_as_a_dark_spot(self):
        cv2.line(self.image, (30, 90), (150, 100), (35, 55, 75), 2)

        count, _, regions = self.analyzer.detect_dark_spots(self.image, self.skin_mask)

        self.assertEqual(count, 0)
        self.assertEqual(regions, [])

    def test_skin_segmentation_removes_thin_dark_hair(self):
        image = np.full((240, 200, 3), (100, 145, 195), dtype=np.uint8)
        cv2.line(image, (40, 150), (160, 150), (25, 35, 45), 2)

        mask, _ = ImageProcessor().get_skin_mask(image, (20, 20, 160, 200))

        self.assertEqual(mask[150, 100], 0)
        self.assertGreater(mask[140, 100], 0)

    def test_skin_mask_keeps_forehead_cheeks_and_chin_in_scope(self):
        image = np.full((240, 200, 3), (100, 145, 195), dtype=np.uint8)
        mask, _ = ImageProcessor().get_skin_mask(image, (20, 20, 160, 200))

        self.assertGreater(mask[45, 100], 0)
        self.assertGreater(mask[130, 50], 0)
        self.assertGreater(mask[205, 100], 0)


if __name__ == "__main__":
    unittest.main()
