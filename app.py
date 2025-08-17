# app.py
from flask import Flask, request, jsonify, render_template, send_file
from ultralytics import YOLO
from PIL import Image
import io, os
import numpy as np

MODEL_PATH = "weights/best.pt"
CLASSES_TXT = "weights/classes.txt"
PORT = int(os.environ.get("PORT", 7860))

PREDICT_KW = dict(
    conf=0.60,      # filter weak detections
    iou=0.50,
    imgsz=512,      # match your training size
    max_det=5,
    augment=True,
    agnostic_nms=False,
    verbose=False,
)

app = Flask(__name__, template_folder="templates")

# --- Load model ---
model = YOLO(MODEL_PATH)
try:
    model.to("cuda")
except Exception:
    model.to("cpu")

# --- Load class names from classes.txt (enforced order) ---
def load_class_names(path):
    if not os.path.exists(path):
        # Fallback to model.names if file missing
        # but your request is to use classes.txt, so better to raise
        raise FileNotFoundError(f"{path} not found. Please create it with one class name per line.")
    with open(path, "r", encoding="utf-8") as f:
        names = [ln.strip() for ln in f if ln.strip()]
    # Build id->name dict: 0..N-1
    return {i: n for i, n in enumerate(names)}

CLASS_NAMES = load_class_names(CLASSES_TXT)

# --- Helpers ---
def _pil_from_upload(fs):
    data = fs.read()
    return Image.open(io.BytesIO(data)).convert("RGB")

def _postprocess_result(r, single_object=False):
    boxes = r.boxes
    if boxes is None or len(boxes) == 0:
        return []

    if single_object:
        top_idx = int(boxes.conf.argmax())
        boxes = boxes[[top_idx]]

    xyxy = boxes.xyxy.cpu().numpy()
    cls = boxes.cls.cpu().numpy().astype(int)
    conf = boxes.conf.cpu().numpy()

    out = []
    for b, c, s in zip(xyxy, cls, conf):
        name = CLASS_NAMES.get(int(c), f"class_{int(c)}")
        out.append({
            "bbox": [float(x) for x in b],
            "class_id": int(c),
            "class_name": name,
            "confidence": float(s),
        })
    return out

def _annotated_png(r):
    plotted = r.plot()             # BGR numpy
    img_rgb = plotted[:, :, ::-1]  # -> RGB
    pil_img = Image.fromarray(img_rgb)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)
    return buf

# --- Routes ---
@app.get("/")
def index():
    return render_template("index.html")

@app.post("/predict")
def predict():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        single_object = bool(request.form.get("single")) or request.args.get("single") == "true"
        return_image = bool(request.form.get("return_image")) or request.args.get("return_image") == "true"

        img = _pil_from_upload(request.files["file"])
        r = model.predict(img, **PREDICT_KW)[0]

        detections = _postprocess_result(r, single_object=single_object)

        if return_image:
            # if single_object requested for image, redraw on filtered result
            if single_object and len(r.boxes) > 0:
                top = int(r.boxes.conf.argmax())
                r.boxes = r.boxes[[top]]
            return send_file(_annotated_png(r), mimetype="image/png")

        return jsonify({"detections": detections})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.get("/health")
def health():
    return jsonify({"status": "ok", "classes": CLASS_NAMES})

if __name__ == "__main__":
    port = int(os.environ.get("PORT",7860))
    app.run(host="0.0.0.0", port=PORT, debug=True, threaded=True)

