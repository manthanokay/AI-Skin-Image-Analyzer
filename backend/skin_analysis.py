import cv2
import numpy as np

class SkinAnalyzer:
    def __init__(self):
        pass

    @staticmethod
    def _compact_shape(cnt):
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        x, y, width, height = cv2.boundingRect(cnt)
        aspect_ratio = max(width, height) / min(width, height) if min(width, height) else float("inf")
        circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter else 0
        hull_area = cv2.contourArea(cv2.convexHull(cnt))
        solidity = area / hull_area if hull_area else 0
        return area, aspect_ratio, circularity, solidity

    def analyze_redness(self, img, skin_mask):
        """
        Calculates redness using a Local Color Deviation approach.
        Each pixel is compared to a locally blurred baseline of the same region
        to eliminate errors from lighting gradients and natural skin tone.
        """
        # 1. Convert to LAB and HSV
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a_chan, b = cv2.split(lab)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        _, sat, _ = cv2.split(hsv)

        # 2. Generate Local Baseline (Spatially-aware Normal Skin Tone)
        # Use a large median blur to create a 'reference' skin tone map.
        # This removes small red spots but keeps the general skin tone/lighting gradient.
        local_baseline_a = cv2.medianBlur(a_chan, 51)

        # 3. Calculate Local Deviation (Current a* - Local Baseline a*)
        # Positive value means the pixel is redder than its immediate neighborhood.
        deviation = a_chan.astype(np.float32) - local_baseline_a.astype(np.float32)

        # 4. Saturation Weighting & Validity
        # Ignore areas with very low saturation (shadows/grey areas)
        # Saturation normalized to 0.0 - 1.0
        sat_norm = sat.astype(np.float32) / 255.0
        weighted_deviation = deviation * sat_norm

        # 5. Mask and Threshold
        # Use a conservative threshold to define 'significant' redness
        sensitivity = 12.0 # Adjustable threshold for a* channel deviation
        redness_mask = (weighted_deviation > sensitivity).astype(np.uint8) * 255
        redness_mask = cv2.bitwise_and(redness_mask, skin_mask)

        # 6. Heatmap Generation
        # Create a smooth gradient from the deviation map
        # Clip the deviation to a reasonable range (e.g., 0 to 40)
        heatmap_raw = np.clip(weighted_deviation, 0, 40)
        heatmap_normalized = (heatmap_raw / 40.0 * 255).astype(np.uint8)

        # Apply Gaussian Blur to the heatmap to remove blockiness and make it 'glow'
        heatmap_smooth = cv2.GaussianBlur(heatmap_normalized, (31, 31), 0)

        # Final mask to ensure heatmap only appears on valid skin
        final_heatmap = cv2.bitwise_and(heatmap_smooth, skin_mask)

        # 7. Overall Redness Score
        # Percentage of valid skin that exceeds the sensitivity threshold
        skin_area = np.sum(skin_mask > 0)
        if skin_area == 0:
            return 0, np.zeros(img.shape[:2], dtype=np.uint8)

        red_area = np.sum(redness_mask > 0)
        overall_score = np.clip(red_area / skin_area, 0, 1)

        return overall_score, final_heatmap



    def detect_dark_spots(self, img, skin_mask):
        """
        Identifies dark spots using adaptive luminance analysis and
        strict shape validation to filter out hair/stubble.
        """
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_channel, a, b = cv2.split(lab)

        local_mean = cv2.GaussianBlur(l_channel, (25, 25), 0)
        diff = local_mean.astype(np.int16) - l_channel.astype(np.int16)

        dark_mask = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)[1].astype(np.uint8)
        dark_mask = cv2.bitwise_and(dark_mask, skin_mask)

        kernel = np.ones((3, 3), np.uint8)
        dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        spots = []
        for cnt in contours:
            area, aspect_ratio, circularity, solidity = self._compact_shape(cnt)
            _, _, width, height = cv2.boundingRect(cnt)
            if 8 < area < 500:
                if min(width, height) < 4 or aspect_ratio > 1.8 or circularity < 0.42 or solidity < 0.78:
                    continue

                mask_roi = np.zeros(img.shape[:2], dtype=np.uint8)
                cv2.drawContours(mask_roi, [cnt], -1, 255, -1)
                ring = cv2.dilate(mask_roi, np.ones((7, 7), np.uint8)) - mask_roi
                ring = cv2.bitwise_and(ring, skin_mask)

                if not np.any(ring):
                    continue
                avg_inner = np.mean(l_channel[mask_roi > 0])
                avg_outer = np.mean(l_channel[ring > 0])

                if avg_inner > avg_outer - 18:
                    continue

                spots.append(cnt)

        return len(spots), dark_mask, spots


    def detect_acne_like(self, img, skin_mask):
        """
        Finds small, compact red or dark spots that contrast with nearby skin.
        """
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, _ = cv2.split(lab)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        _, saturation, _ = cv2.split(hsv)

        # Compare each pixel with its local skin-tone baseline so detection works
        # across different skin tones and does not depend on a fixed red hue alone.
        baseline_a = cv2.GaussianBlur(a_channel, (15, 15), 0)
        baseline_l = cv2.GaussianBlur(l_channel, (15, 15), 0)
        red_deviation = a_channel.astype(np.int16) - baseline_a.astype(np.int16)
        dark_deviation = baseline_l.astype(np.int16) - l_channel.astype(np.int16)

        red_candidates = (red_deviation >= 7) & (saturation >= 45)
        dark_candidates = (dark_deviation >= 22) & (saturation >= 35)
        mask = (red_candidates | dark_candidates).astype(np.uint8) * 255
        mask = cv2.bitwise_and(mask, skin_mask)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        regions = []
        for cnt in contours:
            area, aspect_ratio, circularity, solidity = self._compact_shape(cnt)
            _, _, width, height = cv2.boundingRect(cnt)
            if 8 <= area <= 500:
                if min(width, height) < 4 or aspect_ratio > 1.8 or circularity < 0.42 or solidity < 0.78:
                    continue

                component_mask = np.zeros(img.shape[:2], dtype=np.uint8)
                cv2.drawContours(component_mask, [cnt], -1, 255, -1)
                ring = cv2.dilate(component_mask, np.ones((7, 7), np.uint8)) - component_mask
                ring = cv2.bitwise_and(ring, skin_mask)
                if not np.any(ring):
                    continue
                region_redness = np.mean(red_deviation[component_mask > 0])
                region_darkness = np.mean(dark_deviation[component_mask > 0])
                ring_redness = np.mean(red_deviation[ring > 0])
                ring_darkness = np.mean(dark_deviation[ring > 0])
                if region_redness - ring_redness < 5 and region_darkness - ring_darkness < 16:
                    continue

                regions.append(cnt)

        return len(regions), mask, regions

    def detect_pores(self, img, skin_mask):
        """
        Detects small, dark, circular regions.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
        abs_lap = np.absolute(laplacian).astype(np.uint8)

        _, thresh = cv2.threshold(abs_lap, 40, 255, cv2.THRESH_BINARY)
        pore_mask = cv2.bitwise_and(thresh, skin_mask)

        contours, _ = cv2.findContours(pore_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        pores = []
        for cnt in contours:
            area, aspect_ratio, circularity, solidity = self._compact_shape(cnt)
            if 2 < area < 50 and aspect_ratio <= 1.8 and circularity >= 0.35 and solidity >= 0.7:
                pores.append(cnt)

        return len(pores), pore_mask, pores

    def analyze_oiliness(self, img, skin_mask):
        """
        Detects specular highlights (shiny regions) to estimate oiliness.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # High intensity regions in skin areas are often oil/highlights
        _, shiny_mask = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)
        shiny_mask = cv2.bitwise_and(shiny_mask, skin_mask)

        shiny_area = np.sum(shiny_mask > 0)
        skin_area = np.sum(skin_mask > 0)
        score = shiny_area / skin_area if skin_area > 0 else 0
        return score, shiny_mask

    def analyze_texture_and_uniformity(self, img, skin_mask):
        """
        Analyzes skin tone uniformity and texture.
        """
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_channel, _, _ = cv2.split(lab)

        # Get L values only for skin regions
        skin_pixels = l_channel[skin_mask > 0]
        if len(skin_pixels) == 0:
            return 100, 0

        # Uniformity = 100 - (normalized standard deviation)
        std_dev = np.std(skin_pixels)
        uniformity = max(0, min(100, 100 - (std_dev / 2)))

        # Texture via Laplacian variance
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        masked_gray = cv2.bitwise_and(gray, gray, mask=skin_mask)
        texture_var = cv2.Laplacian(masked_gray, cv2.CV_64F).var()
        smoothness = max(0, min(100, 100 - (texture_var / 10)))

        return uniformity, smoothness
