import os
import cv2
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
from backend.image_processor import ImageProcessor
from backend.skin_analysis import SkinAnalyzer
from backend.ai_analyzer import AIAnalyzer, AIServiceError
from backend.product_recommender import ProductRecommender
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

processor = ImageProcessor()
analyzer = SkinAnalyzer()
ai_engine = AIAnalyzer()
recommender = ProductRecommender()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/analyze', methods=['POST'])
def analyze():
    if 'image' not in request.files:
        return jsonify({"success": False, "error": "No image uploaded"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"success": False, "error": "No selected file"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        # 1. Preprocessing & Face Detection
        original, enhanced, face_box = processor.preprocess(filepath)
        if original is None:
            return jsonify({"success": False, "error": "Invalid image file"}), 400

        if face_box is None:
            return jsonify({
                "success": False,
                "error": "No face detected. Try a clear, well-lit, front-facing photo."
            }), 422

        # 2. Refined Skin Segmentation
        skin_mask, _ = processor.get_skin_mask(enhanced, face_box)

        # 3. Enhanced Feature Analysis
        redness_score, redness_heatmap = analyzer.analyze_redness(enhanced, skin_mask)
        dark_spot_count, dark_mask, dark_spots = analyzer.detect_dark_spots(enhanced, skin_mask)
        acne_count, acne_mask, acne_regions = analyzer.detect_acne_like(enhanced, skin_mask)
        pore_count, pore_mask, pore_regions = analyzer.detect_pores(enhanced, skin_mask)
        oiliness_score, oil_mask = analyzer.analyze_oiliness(enhanced, skin_mask)
        uniformity, smoothness = analyzer.analyze_texture_and_uniformity(enhanced, skin_mask)

        # 4. Generate Annotated Overlay
        overlay = enhanced.copy()

        # REDNESS HEATMAP OVERLAY
        # Convert the normalized 0-255 heatmap to a Red-White gradient
        # Using COLORMAP_JET (Blue-Green-Yellow-Red)
        red_color_map = cv2.applyColorMap(redness_heatmap, cv2.COLORMAP_JET)
        # To make it "Red-only", we mask out the blue/green parts (low values)
        # Create a mask for significant redness (e.g., values > 100)
        significance_mask = cv2.threshold(redness_heatmap, 100, 255, cv2.THRESH_BINARY)[1]
        # Combine with skin mask to ensure it only appears on skin
        significance_mask = cv2.bitwise_and(significance_mask, skin_mask)

        # Blend the color map with the image
        red_overlay = cv2.bitwise_and(red_color_map, red_color_map, mask=significance_mask)
        overlay = cv2.addWeighted(overlay, 1.0, red_overlay, 0.4, 0)

        # Dark spots - yellow circles
        for cnt in dark_spots:
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            cv2.circle(overlay, (int(x), int(y)), int(radius), (0, 255, 255), 2)

        # Pimple-like candidates - blue circles
        for cnt in acne_regions:
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            cv2.circle(overlay, (int(x), int(y)), max(3, int(radius)), (255, 0, 0), 2)


        # Save results
        annotated_path = os.path.join(app.config['UPLOAD_FOLDER'], f"res_{filename}")
        cv2.imwrite(annotated_path, overlay)

        # 5. Metrics & Report Data
        dip_metrics = {
            "redness_score": round(redness_score, 2),
            "dark_spots": dark_spot_count,
            "acne_like_regions": acne_count,
            "pore_like_regions": pore_count,
            "oiliness_score": round(oiliness_score, 2),
            "skin_uniformity": int(uniformity),
            "skin_smoothness": int(smoothness),
            "face_detected": True,
            "image_quality": 85
        }

        # Determine attention level
        attention = "LOW"
        if dip_metrics['acne_like_regions'] > 10 or dip_metrics['redness_score'] > 0.3:
            attention = "HIGH — Consider professional evaluation"
        elif dip_metrics['dark_spots'] > 5 or dip_metrics['acne_like_regions'] > 3:
            attention = "MODERATE"

        ai_result = ai_engine.analyze_skin_with_ai(dip_metrics)
        products = recommender.recommend(dip_metrics)
        routine = recommender.generate_routine(dip_metrics)

        return jsonify({
            "success": True,
            "metrics": dip_metrics,
            "attention_level": attention,
            "annotated_image": f"/uploads/{f'res_{filename}'}",
            "ai_interpretation": ai_result.get("interpretation"),
            "recommendations": products,
            "routine": routine
        })

    except Exception as e:
        app.logger.exception("Error during image analysis")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"success": False, "error": "A JSON request body is required."}), 400

    message = data.get("message")
    if not isinstance(message, str) or not message.strip():
        return jsonify({"success": False, "error": "Enter a message to continue."}), 400
    if len(message) > 2000:
        return jsonify({"success": False, "error": "Messages must be 2,000 characters or fewer."}), 400

    history = data.get("history", [])
    if not isinstance(history, list) or len(history) > 10:
        return jsonify({"success": False, "error": "Chat history must contain at most 10 messages."}), 400
    validated_history = []
    for item in history:
        if (
            not isinstance(item, dict)
            or item.get("role") not in ("user", "assistant")
            or not isinstance(item.get("content"), str)
            or not item["content"].strip()
            or len(item["content"]) > 2000
        ):
            return jsonify({"success": False, "error": "Chat history contains an invalid message."}), 400
        validated_history.append({
            "role": item["role"],
            "content": item["content"].strip()
        })

    metrics = data.get("skin_context")
    skin_context = None
    if isinstance(metrics, dict):
        allowed_metrics = (
            "acne_like_regions", "redness_score", "dark_spots",
            "pore_like_regions", "skin_smoothness"
        )
        skin_context = {
            key: metrics[key]
            for key in allowed_metrics
            if key in metrics and isinstance(metrics[key], (int, float))
        }

    try:
        reply = ai_engine.chat(message.strip(), validated_history, skin_context)
        return jsonify({"success": True, "reply": reply})
    except AIServiceError as e:
        app.logger.warning("Assistant chat unavailable: %s", e)
        return jsonify({"success": False, "error": str(e)}), 503

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
