# AI Skin Image Analyzer

A professional web application for analyzing visible skin characteristics using Digital Image Processing (DIP) and AI-powered educational interpretation.

## 🚀 Features
- **Image Upload & Live Capture:** Upload images or take a photo directly from the browser.
- **Face Detection:** Bundled YuNet model detects faces locally before analysis.
- **DIP Pipeline:** Face-constrained skin segmentation and shape-filtered local-contrast candidates.
- **Feature Analysis:** Estimates redness, dark spots, pimple-like regions, and pores.
- **AI Assistant:** Uses a configurable chat-completions API to explain results and answer questions.
- **Product Recommendations:** Suggests general skincare categories based on results.
- **Crystal UI:** A premium, modern glassmorphism interface.

## 🛠️ Technologies
- **Backend:** Python, Flask, OpenCV, NumPy, Pillow, scikit-image.
- **Frontend:** HTML5, CSS3, JavaScript.
- **AI:** OpenAI-compatible chat-completions API.

## 📦 Installation

1. **Clone the repository**
2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # Linux/Mac
   source venv/bin/activate
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Set up environment variables:**
   - Copy `.env.example` to `.env`
   - Set `AI_API_KEY`, `AI_API_URL` (the chat-completions endpoint), and optionally `AI_MODEL`.

## 🏃 Running the App
```bash
python app.py
```
Open `http://127.0.0.1:5000` in your browser.

## 🎓 Academic Context (How it Works)
The project implements a full DIP pipeline:
1. **Preprocessing:** Normalizes image size and enhances contrast using CLAHE.
2. **Segmentation:** Uses HSV and YCrCb color spaces to isolate skin pixels.
3. **Feature Detection:**
   - Redness: Analyzed via color channel ratios.
   - Dark Spots: Found using luminance thresholding in LAB space.
   - Pimple-like regions: Local red and dark contrast candidates are grouped and shape-filtered.
   - Texture: Estimated via Laplacian variance.
4. **Annotation:** Generates a result overlay using contour detection and color-coding.

YuNet is bundled from [OpenCV Zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet)
under the MIT license; its license is included next to the model. Face blemish counts are
image-processing estimates, not predictions from a trained acne-classification model. Lighting,
camera quality, makeup, hair, and skin tone can affect results; use a clear, front-facing photo.
The assistant appears after a face photo is analyzed and requires a configured AI API. All analysis
and assistant responses are educational only and do NOT provide medical diagnoses.
