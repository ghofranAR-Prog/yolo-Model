import os, io, gc, logging
from typing import List, Dict, Any
from flask import Flask, request, jsonify, render_template
from PIL import Image, UnidentifiedImageError
import torch

# ---- Keep PyTorch light on CPU to reduce RAM usage ----
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
torch.set_num_threads(1)

# Optional: Hugging Face Hub (only used if HF_REPO is set)
try:
    from huggingface_hub import hf_hub_download  # noqa
    HF_AVAILABLE = True
except Exception:
    HF_AVAILABLE = False

from ultralytics import YOLO

app = Flask(__name__, template_folder="templates", static_folder="static")

# ---- Request limits (avoid big files) ----
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # 5MB uploads

# ---- Model / classes loading ----
PORT = int(os.environ.get("PORT", 7860))
DEVICE = "cpu"

# You can set these in Render env vars if you host weights on HF:
HF_REPO = os.environ.get("HF_REPO", "").strip()         # e.g. "yourname/yolo-weights"
HF_WEIGHTS = os.environ.get("HF_WEIGHTS", "best.pt")
HF_CLASSES = os.environ.get("HF_CLASSES", "classes.txt")

LOCAL_WEIGHTS = os.environ.get("MODEL_PATH", "weights/best.pt")
LOCAL_CLASSES = os.environ.get("CLASSES_FILE", "weights/classes.txt")

MODEL = None
CLASS_NAMES: List[str] | Dict[int, str] | None = None
MODEL_LOAD_ERROR = None


def _load_model_and_classes():
    """Load YOLO model once, on CPU, with optional HF fallback."""
    global MODEL, CLASS_NAMES, MODEL_LOAD_ERROR
    try:
        weights_path = None
        classes_path = None

        if HF_REPO and HF_AVAILABLE:
            # Download from HF Hub to cache
            weights_path = hf_hub_download(repo_id=HF_REPO, filename=HF_WEIGHTS)
            if HF_CLASSES:
                classes_path = hf_hub_download(repo_id=HF_REPO, filename=HF_CLASSES)
        elif os.path.exists(LOCAL_WEIGHTS):
            weights_path = LOCAL_WEIGHTS
            if os.path.exists(LOCAL_CLASSES):
                classes_path = LOCAL_CLASSES

        # Final fallback to public nano model if nothing else is found
        if not weights_path:
            weights_path = "yolov8n.pt"  # ultralytics will fetch automatically

        MODEL = YOLO(weights_path)
        # force CPU
        MODEL.to(DEVICE)

        # Load class names
        if classes_path and os.path.exists(classes_path):
            with open(classes_path, "r", encoding="utf-8") as f:
                CLASS_NAMES = [ln.strip() for ln in f if ln.strip()]
        else:
            # use model's internal names
            # MODEL.names may be list or dict depending on version
            names = MODEL.names
            CLASS_NAMES = list(names.values()) if isinstance(names, dict) else names

        MODEL_LOAD_ERROR = None
    except Exception as e:
        MODEL = None
        CLASS_NAMES = None
        MODEL_LOAD_ERROR = str(e)


_load_model_and_classes()


@app.errorhandler(Exception)
def handle_unexpected(e):
    logging.exception("Unhandled error")
    return jsonify({"status": "error", "message": str(e)}), 500


@app.get("/")
def home():
    # optional: serve your index.html if present
    try:
        return render_template("index.html")
    except Exception:
        return jsonify({"status": "ok", "message": "Service running"})


@app.get("/health")
def health():
    if MODEL_LOAD_ERROR:
        return jsonify({"status": "error", "detail": MODEL_LOAD_ERROR}), 500
    return jsonify({"status": "ok", "classes": CLASS_NAMES})


@app.post("/predict")
def predict():
    """
    Accepts multipart/form-data with field 'image'.
    Returns JSON with detections (class, score, box).
    RAM-friendly: resizes to max 320px.
    """
    if MODEL_LOAD_ERROR or MODEL is None:
        return jsonify({"status": "error", "message": f"model not loaded: {MODEL_LOAD_ERROR}"}), 500

    if "image" not in request.files:
        return jsonify({"status": "error", "message": "No file field 'image' in form-data"}), 400

    file = request.files["image"]
    if not file or file.filename == "":
        return jsonify({"status": "error", "message": "Empty or missing filename"}), 400

    try:
        # Read as PIL; this keeps memory low and avoids cv2 overhead
        img = Image.open(file.stream).convert("RGB")

        # Downscale large images to reduce RAM usage
        MAX_SIDE = 320  # keep small on free tier
        img.thumbnail((MAX_SIDE, MAX_SIDE))

        # Inference on CPU, small imgsz
        with torch.no_grad():
            results = MODEL.predict(img, device=DEVICE, imgsz=MAX_SIDE, conf=0.25, verbose=False)

        dets = []
        r = results[0]
        if hasattr(r, "boxes") and r.boxes is not None and len(r.boxes) > 0:
            xyxy = r.boxes.xyxy.cpu().tolist()
            conf = r.boxes.conf.cpu().tolist()
            cls = r.boxes.cls.cpu().tolist()

            # Normalize CLASS_NAMES access
            def name_for(cid: int):
                if isinstance(CLASS_NAMES, dict):
                    return CLASS_NAMES.get(cid)
                if isinstance(CLASS_NAMES, list) and 0 <= cid < len(CLASS_NAMES):
                    return CLASS_NAMES[cid]
                return str(cid)

            for (x1, y1, x2, y2), c, p in zip(xyxy, cls, conf):
                cid = int(c)
                dets.append({
                    "class_id": cid,
                    "class_name": name_for(cid),
                    "confidence": float(p),
                    "box": [float(x1), float(y1), float(x2), float(y2)]
                })

        return jsonify({
            "status": "ok",
            "num_detections": len(dets),
            "detections": dets,
            "image_size": {"w": img.width, "h": img.height}
        })

    except UnidentifiedImageError:
        return jsonify({"status": "error", "message": "Not a valid image"}), 400
    except Exception as e:
        logging.exception("Predict error")
        return jsonify({"status": "error", "message": str(e)}), 500
    finally:
        # Aggressive cleanup to keep memory below 512MB
        try:
            del img
        except Exception:
            pass
        try:
            del results
        except Exception:
            pass
        gc.collect()
        # (No CUDA on Render free tier, but keep for compatibility)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=False, threaded=True)
