from pathlib import Path
import threading

import cv2
import numpy as np

class ImageProcessor:
    def __init__(self):
        self.target_width = 800
        model_path = Path(__file__).resolve().parent / "models" / "face_detection_yunet_2023mar.onnx"
        if not model_path.is_file():
            raise FileNotFoundError(f"YuNet face detection model not found: {model_path}")
        self.face_detector = cv2.FaceDetectorYN_create(
            str(model_path),
            "",
            (self.target_width, self.target_width),
            0.65,
            0.3,
            5000
        )
        self.face_detector_lock = threading.Lock()

    def _detect_face(self, img):
        height, width = img.shape[:2]
        with self.face_detector_lock:
            self.face_detector.setInputSize((width, height))
            _, detections = self.face_detector.detect(img)

        if detections is None or len(detections) == 0:
            return None

        candidates = [face for face in detections if face[14] >= 0.65]
        if not candidates:
            return None
        face = max(candidates, key=lambda item: item[2] * item[3])
        x, y, w, h = face[:4]
        left = max(0, int(round(x)))
        top = max(0, int(round(y)))
        right = min(width, int(round(x + w)))
        bottom = min(height, int(round(y + h)))
        if right <= left or bottom <= top:
            return None
        return left, top, right - left, bottom - top

    def preprocess(self, image_path):
        """
        Standardizes the image and detects the face to constrain analysis.
        """
        img = cv2.imread(image_path)
        if img is None:
            return None, None, None

        # Resize while preserving aspect ratio
        h, w = img.shape[:2]
        aspect_ratio = h / w
        new_h = int(self.target_width * aspect_ratio)
        img = cv2.resize(img, (self.target_width, new_h))

        face_box = self._detect_face(img)

        # Contrast Enhancement using CLAHE
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

        return img, enhanced, face_box

    def get_skin_mask(self, img, face_box=None):
        """
        Segments the face while excluding the eyes, mouth, hair, and background.
        Returns: (final_mask, debug_masks_dict)
        """
        # 1. Base Skin Masking (HSV + YCrCb)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_skin_hsv = np.array([0, 10, 35], dtype=np.uint8)
        upper_skin_hsv = np.array([35, 255, 255], dtype=np.uint8)
        mask_hsv = cv2.inRange(hsv, lower_skin_hsv, upper_skin_hsv)

        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        lower_skin_ycrcb = np.array([0, 125, 65], dtype=np.uint8)
        upper_skin_ycrcb = np.array([255, 185, 145], dtype=np.uint8)
        mask_ycrcb = cv2.inRange(ycrcb, lower_skin_ycrcb, upper_skin_ycrcb)

        base_mask = cv2.bitwise_or(mask_hsv, mask_ycrcb)

        # 2. Face Box & Feature Exclusion Mask
        exclusion_mask = np.ones(img.shape[:2], dtype=np.uint8) * 255

        if face_box is not None:
            x, y, w, h = face_box
            # Keep the full forehead, cheeks, nose, and chin inside the analysis area.
            face_roi = np.zeros(exclusion_mask.shape, dtype=np.uint8)
            cv2.ellipse(
                face_roi,
                (x + w // 2, y + int(h * 0.53)),
                (max(1, int(w * 0.46)), max(1, int(h * 0.45))),
                0,
                0,
                360,
                255,
                -1
            )
            exclusion_mask = cv2.bitwise_and(exclusion_mask, face_roi)

            # Exclude only the main facial features; keep the surrounding skin in scope.
            for eye_x in (x + int(w * 0.32), x + int(w * 0.68)):
                cv2.ellipse(
                    exclusion_mask,
                    (eye_x, y + int(h * 0.36)),
                    (max(1, int(w * 0.16)), max(1, int(h * 0.065))),
                    0,
                    0,
                    360,
                    0,
                    -1
                )
            cv2.ellipse(
                exclusion_mask,
                (x + w // 2, y + int(h * 0.76)),
                (max(1, int(w * 0.20)), max(1, int(h * 0.075))),
                0,
                0,
                360,
                0,
                -1
            )
        else:
            exclusion_mask = np.zeros(exclusion_mask.shape, dtype=np.uint8)

        # 3. Linear Hair Detection (Morphological)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, dark_pixels = cv2.threshold(gray, 105, 255, cv2.THRESH_BINARY_INV)
        hair_candidates = cv2.bitwise_and(dark_pixels, base_mask)
        line_length = 9
        line_kernels = (
            np.eye(line_length, dtype=np.uint8),
            np.fliplr(np.eye(line_length, dtype=np.uint8)),
            np.ones((1, line_length), dtype=np.uint8),
            np.ones((line_length, 1), dtype=np.uint8)
        )
        hair_mask = np.zeros(img.shape[:2], dtype=np.uint8)
        for line_kernel in line_kernels:
            line_hair = cv2.morphologyEx(hair_candidates, cv2.MORPH_OPEN, line_kernel)
            hair_mask = cv2.bitwise_or(hair_mask, line_hair)
        hair_mask = cv2.dilate(hair_mask, np.ones((3, 3), np.uint8), iterations=1)

        # 4. Final Skin ROI Calculation
        final_mask = cv2.bitwise_and(base_mask, exclusion_mask)
        final_mask = cv2.bitwise_and(final_mask, cv2.bitwise_not(hair_mask))

        # Final cleanup
        kernel = np.ones((3, 3), np.uint8)
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel)

        # Debugging dictionary
        debug_masks = {
            "base_skin": base_mask,
            "exclusion": exclusion_mask,
            "hair": hair_mask,
            "final_roi": final_mask
        }

        return final_mask, debug_masks

    def get_edge_map(self, img):
        """
        Generates a Canny edge map.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(blurred, 50, 150)
        return edges
