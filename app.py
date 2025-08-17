# app.py
import os, io, gc, logging, base64
from uuid import uuid4
from typing import List, Dict, Optional, Union
from flask import Flask, request, jsonify, render_template
from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge, HTTPException
import torch

# ---------- Low-RAM settings ----------
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
torch.set_num_threads(1)

PORT   = int(os.environ.get("PORT", 7860))
DEVICE = "cpu"

# Keep these small on 512MB tiers
MAX_SIDE = int(os.environ.get("MAX_SIDE", 320))     # inference input size cap
MAX_DET  = int(os.environ.get("MAX_DET", 15))       # max detections to parse
MAX_ANN  = int(os.environ.get("MAX_ANN", 512))      # annotated image max side

# Paths (change via env if you like)
LOCAL_WEIGHTS = os.environ.get("MODEL_PATH", "weights/best.pt")
LOCAL_CLASSES = os.environ.get("CLASSES_FILE", "weights/classes.txt")

# Upload cap (MB). 413 happens before route code; keep this high enough.
UPLOAD_MAX_MB = int(os.environ.get("UPLOAD_MAX_MB", "10"))

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = UPLOAD_MAX_MB * 1024 * 1024  # e.g., 10 MB

# Globals
MODEL: Optional["YOLO"] = None
CLASS_NAMES: Optional[Union[List[str], Dict[int, str]]] = None
MODEL_LOAD_ERROR: Optional[str] = None
WEIGHTS_IN_USE: Optional[str] = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ---------- Error handlers (return JSON, not HTML) ----------
@app.errorhandler(RequestEntityTooLarge)
def handle_413(e):
    return jsonify({"status": "error", "code": 413,
                    "message": f"File too large. Max {UPLOAD_MAX_MB} MB."}), 413

@app.errorhandler(HTTPException)
def handle_http_exception(e: HTTPException):
    # Any other HTTP error (404, 405, etc.)
    return jsonify({"status": "error", "code": e.code or 500, "message": e.description}), e.code or 500

@app.errorhandler(Exception)
def handle_500(e):
    logging.exception("Unhandled error")
    return jsonify({"status": "error", "code": 500, "message": str(e)}), 500


# ---------- Helpers ----------
def _try_load_classes(path: str) -> Optional[List[str]]:
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                lines = [ln.strip() for ln in f if ln.strip()]
                return lines or None
    except Exception:
        pass
    return None


def _load_model_once() -> None:
    """Load YOLO on CPU with fallbacks and record class names."""
    global MODEL, CLASS_NAMES, MODEL_LOAD_ERROR, WEIGHTS_IN_USE
    try:
        from ultralytics import YOLO  # import here to keep startup light
    except Exception as e:
        MODEL_LOAD_ERROR = f"Ultralytics not importable: {e}"
        logging.error("[BOOT] %s", MODEL_LOAD_ERROR)
        return

    candidates = []
    if os.path.exists(LOCAL_WEIGHTS):
        candidates.append(LOCAL_WEIGHTS)
    candidates.append("yolov8n.pt")  # public fallback

    last_err = None
    for w in candidates:
        try:
            m = YOLO(w)
            m.to(DEVICE)

            names = getattr(m, "names", None)
            if isinstance(names, dict):
                max_idx = max(names.keys()) if names else -1
                classes = [names.get(i, str(i)) for i in range(max_idx + 1)]
            else:
                classes = list(names) if names is not None else []

            override = _try_load_classes(LOCAL_CLASSES)
            if override:
                classes = override

            MODEL, CLASS_NAMES, WEIGHTS_IN_USE = m, classes, w
            MODEL_LOAD_ERROR = None
            logging.info("[BOOT] loaded weights: %s", w)
            logging.info("[BOOT] classes: %s", classes)
            return
        except Exception as e:
            last_err = str(e)
            logging.warning("[BOOT] failed to load %s -> %s", w, last_err)

    MODEL = None
    CLASS_NAMES = None
    WEIGHTS_IN_USE = None
    MODEL_LOAD_ERROR = last_err or "Unknown model load error"
    logging.error("[BOOT] failed to load model: %s", MODEL_LOAD_ERROR)


_load_model_once()


def _text_size(draw: ImageDraw.ImageDraw, text: str, font: Optional[ImageFont.ImageFont]) -> tuple[int, int]:
    try:
        box = draw.textbbox((0, 0), text, font=font)  # (l, t, r, b)
        return (box[2] - box[0], box[3] - box[1])
    except Exception:
        return draw.textsize(text, font=font)


def _draw_annotations(pil_img: Image.Image, dets: List[dict]) -> Image.Image:
    img = pil_img.copy()
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for d in dets:
        x1, y1, x2, y2 = [float(v) for v in d["box"]]
        cls_name = d.get("class_name", str(d.get("class_id", "?")))
        conf = float(d.get("confidence", 0.0))
        label = f"{cls_name} {conf:.2f}"

        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        tw, th = _text_size(draw, label, font)
        pad = 2
        bg = [x1, max(0.0, y1 - th - 2 * pad), x1 + tw + 2 * pad, y1]
        draw.rectangle(bg, fill="red")
        draw.text((x1 + pad, y1 - th - pad), label, fill="white", font=font)

    return img


def _encode_png_base64(pil_img: Image.Image) -> str:
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")


def _class_name_from_id(cid: int) -> str:
    cn = CLASS_NAMES
    if isinstance(cn, dict):
        return str(cn.get(cid, cid))
    if isinstance(cn, list) and 0 <= cid < len(cn):
        return str(cn[cid])
    return str(cid)


# ---------- Routes ----------
@app.get("/")
def index():
    try:
        return render_template("index.html")
    except Exception:
        return jsonify({"status": "ok", "message": "Service running"})


@app.get("/health")
def health():
    payload = {
        "status": "ok" if MODEL and not MODEL_LOAD_ERROR else "error",
        "weights_in_use": WEIGHTS_IN_USE,
        "classes": CLASS_NAMES,
        "upload_max_mb": UPLOAD_MAX_MB,
    }
    if MODEL_LOAD_ERROR:
        payload["detail"] = MODEL_LOAD_ERROR
    return jsonify(payload), (200 if payload["status"] == "ok" else 500)


@app.post("/predict")
def predict():
    """
    Accepts multipart/form-data with field 'image' or 'file'.

    Query params:
      - annotate=0|1 (default 1): whether to generate an annotated preview
      - inline=0|1   (default 1): include Base64 for backward compatibility
    """
    if MODEL_LOAD_ERROR or MODEL is None:
        return jsonify({"status": "error", "message": f"model not loaded: {MODEL_LOAD_ERROR}"}), 500

    up = request.files.get("image") or request.files.get("file")
    if not up or up.filename == "":
        return jsonify({"status": "error", "message": "No uploaded file (use field 'image' or 'file')"}), 400

    try:
        annotate_flag = int(request.args.get("annotate", "1"))
    except Exception:
        annotate_flag = 1
    try:
        inline_flag = int(request.args.get("inline", "1"))
    except Exception:
        inline_flag = 1

    annotated_b64 = None
    annotated_url = None
    img = None
    results = None

    try:
        img = Image.open(up.stream).convert("RGB")
        img.thumbnail((MAX_SIDE, MAX_SIDE))

        with torch.no_grad():
            out = MODEL.predict(
                img, device=DEVICE, imgsz=MAX_SIDE, conf=0.15, max_det=MAX_DET, verbose=False
            )
            results = list(out) if isinstance(out, (list, tuple)) else [out]

        dets: List[Dict] = []

        if results and hasattr(results[0], "boxes"):
            r0 = results[0]
            boxes = getattr(r0, "boxes", None)

            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.detach().cpu().tolist()
                conf = boxes.conf.detach().cpu().tolist()
                cls  = boxes.cls.detach().cpu().tolist()

                for (x1, y1, x2, y2), c, p in zip(xyxy, cls, conf):
                    cid = int(c) if isinstance(c, (int, float)) else int(c[0])
                    dets.append({
                        "class_id": int(cid),
                        "class_name": _class_name_from_id(int(cid)),
                        "confidence": float(p),
                        "box": [float(x1), float(y1), float(x2), float(y2)],
                    })

                if annotate_flag == 1:
                    ann = _draw_annotations(img, dets)
                    ann.thumbnail((MAX_ANN, MAX_ANN))

                    out_dir = os.path.join(app.static_folder, "annotated")
                    os.makedirs(out_dir, exist_ok=True)
                    fname = f"{uuid4().hex}.png"
                    save_path = os.path.join(out_dir, secure_filename(fname))
                    ann.save(save_path, format="PNG", optimize=True)
                    annotated_url = f"/static/annotated/{fname}"

                    if inline_flag == 1:
                        annotated_b64 = _encode_png_base64(ann)

        payload = {
            "status": "ok",
            "num_detections": len(dets),
            "detections": dets,
            "image_size": {"w": int(img.width), "h": int(img.height)},
            "weights_in_use": WEIGHTS_IN_USE,
        }
        if annotate_flag == 1:
            payload["annotated_image_url"] = annotated_url
            payload["annotated_image"] = annotated_b64

        return jsonify(payload)

    except UnidentifiedImageError:
        return jsonify({"status": "error", "message": "Not a valid image"}), 400
    except Exception as e:
        logging.exception("Predict error")
        return jsonify({"status": "error", "message": str(e)}), 500
    finally:
        try: del img
        except Exception: pass
        try: del results
        except Exception: pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=False, threaded=True)
