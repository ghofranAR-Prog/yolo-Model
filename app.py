import os
from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO

# --- Config ---
PORT = int(os.environ.get("PORT", 7860))
UPLOAD_DIR = "uploads"
WEIGHTS_PATH = "weights/best.pt"   # change if your file has a different name

os.makedirs(UPLOAD_DIR, exist_ok=True)

app = Flask(__name__, template_folder="templates", static_folder="static")

# Load model once on startup
try:
    model = YOLO(WEIGHTS_PATH)
except Exception as e:
    # Don't crash; expose a clear error via /health
    model = None
    MODEL_LOAD_ERROR = str(e)
else:
    MODEL_LOAD_ERROR = None

# If you have a classes.txt inside weights/, read it
CLASSES_TXT = os.path.join("weights", "classes.txt")
if os.path.exists(CLASSES_TXT):
    with open(CLASSES_TXT, "r", encoding="utf-8") as f:
        CLASS_NAMES = [line.strip() for line in f if line.strip()]
else:
    CLASS_NAMES = None


@app.get("/")
def home():
    return render_template("index.html")


@app.get("/health")
def health():
    if MODEL_LOAD_ERROR:
        return jsonify({"status": "error", "detail": MODEL_LOAD_ERROR}), 500
    return jsonify({"status": "ok", "classes": CLASS_NAMES})


@app.post("/predict")
def predict():
    """
    Expects multipart/form-data with a file field named 'file'.
    Returns JSON:
    {
      "detections": [
        {"class_id": 0, "class_name": "person", "score": 0.91,
         "box": [x1, y1, x2, y2]}
      ],
      "image": {"filename": "..."}
    }
    """
    try:
        if MODEL_LOAD_ERROR or model is None:
            return jsonify({"error": f"model not loaded: {MODEL_LOAD_ERROR}"}), 500

        if "file" not in request.files:
            return jsonify({"error": "no file part 'file' in request"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "empty filename"}), 400

        # Save upload
        save_path = os.path.join(UPLOAD_DIR, file.filename)
        file.save(save_path)

        # Run inference (no saving images; we only need JSON)
        results = model.predict(save_path, verbose=False)

        detections = []
        for r in results:
            # r.boxes: xyxy, conf, cls
            if r.boxes is None:
                continue
            xyxy = r.boxes.xyxy.cpu().tolist()
            conf = r.boxes.conf.cpu().tolist()
            cls = r.boxes.cls.cpu().tolist()

            for i in range(len(cls)):
                x1, y1, x2, y2 = xyxy[i]
                cid = int(cls[i])
                score = float(conf[i])
                cname = (r.names.get(cid) if hasattr(r, "names") else None)
                # Prefer our classes.txt if present
                if CLASS_NAMES and 0 <= cid < len(CLASS_NAMES):
                    cname = CLASS_NAMES[cid]
                detections.append({
                    "class_id": cid,
                    "class_name": cname,
                    "score": round(score, 4),
                    "box": [int(x1), int(y1), int(x2), int(y2)]
                })

        return jsonify({
            "detections": detections,
            "image": {"filename": file.filename}
        })
    except Exception as e:
        # Always JSON, even on errors
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # For local runs only; Render uses gunicorn to serve `app:app`
    app.run(host="0.0.0.0", port=PORT, debug=True, threaded=True)
