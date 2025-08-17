# YOLO Model Deployment 
<docs/PaintBrush.png>
<docs/ScrewDriver.png>

This repository contains a **YOLOv8 object detection model** deployed with **Flask** and packaged with **Docker** for easy containerized deployment.

---

##  Project Overview
- Train and serve a YOLOv8 model for object detection.  
- Expose predictions through a simple Flask web interface.  
- Run locally or inside Docker containers for portability.  

---
* Table of Contents

* Demo

* What this project does

* Repository structure

* Quick start

* Run locally (no Docker)

* Run with Docker (CPU)

* Run with Docker Compose (GPU, optional)

* Environment variables

* Using the interface

* HTTP API

* GET /health

* POST /predict

* Example response

* Known issues & limitations

* Tips for deployment (Render)

* License & credits


### Demo

* Local: http://localhost:7860

Live (Render): https://yolo-model-jdh7.onrender.com

### What this project does

* Serves a YOLOv8 model for object detection (trained on screwdriver vs paintbrush).

* Provides a web UI to upload an image and visualize detections with bounding boxes and labels.

* Exposes a simple REST API for programmatic inference.

* Packaged with Docker for reproducible runs and easy deployment.

##  Repository Structure
yolo-Model/
│-- app.py # Flask application
│-- requirements.txt # Python dependencies
│-- Dockerfile # Build instructions for Docker image
│-- docker-compose.gpu.yml # Optional GPU-enabled setup
│-- templates/ # HTML templates
│-- weights/ # YOLO model weights
│-- .gitignore # Ignored files

## Running the Application

### Local (without Docker)
# 1) Create and activate a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

# 2) Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 3) (Optional) Put your model at weights/best.pt
#    Otherwise the app falls back to yolov8n.pt

# 4) Run
python app.py
# Open http://localhost:7860


### With Docker:

# Build the image
docker build -t yolo-flask:latest .

# Run the container
docker run --rm -p 7860:7860 \
  -e PORT=7860 \
  -e MODEL_PATH="weights/best.pt" \
  yolo-flask:latest
# Open http://localhost:7860

### If you want to use a host model file, mount it:
docker run --rm -p 7860:7860 -e PORT=7860 \
  -v "%cd%/weights:/app/weights" \
  yolo-flask:latest

### Run with Docker Compose (GPU, optional):
docker compose -f docker-compose.gpu.yml up --build


### Environment variables

These can be set in Render, Docker, or your shell:
  
| Var             | Default               | Description                                                   |
| --------------- | --------------------- | ------------------------------------------------------------- |
| `PORT`          | `7860`                | Server port. App binds to `0.0.0.0`.                          |
| `MODEL_PATH`    | `weights/best.pt`     | Path to YOLO weights (falls back to `yolov8n.pt` if missing). |
| `CLASSES_FILE`  | `weights/classes.txt` | Optional list of class names (one per line).                  |
| `MAX_SIDE`      | `320`                 | Inference resize cap (keeps RAM low).                         |
| `MAX_DET`       | `15`                  | Maximum detections to parse.                                  |
| `MAX_ANN`       | `512`                 | Max side for server-rendered annotated preview.               |
| `UPLOAD_MAX_MB` | `10`                  | Upload size cap; JSON error returned if exceeded.             |

### Using the interface:

* Open the app in your browser.

* Click browse (or drag & drop) to select an image.

* Tick Annotated image to show boxes & labels.

* Click Predict to run detection.

* The right panel shows either a JSON result or an annotated image (and a JSON snippet in the Network tab).

###HTTP API
{
  "status": "ok",
  "weights_in_use": "weights/best.pt",
  "classes": ["screwdriver", "paintbrush"],
  "upload_max_mb": 10
}
### Post/Predict:
* Content-Type: multipart/form-data

* Fields: file or image (an image)

* Query params:

* annotate=0|1 (default 1): produce an annotated preview

* inline=0|1 (default 1): include Base64 (annotated_image) for backward compatibility

* URL is also returned at annotated_image_url (small response).
* 
* curl -X POST "http://localhost:7860/predict?annotate=1&inline=1" \
  -F "file=@sample.jpg"

### Response (example)
{
  "status": "ok",
  "num_detections": 2,
  "detections": [
    {
      "class_id": 1,
      "class_name": "screwdriver",
      "confidence": 0.91,
      "box": [33.6, 52.3, 282.9, 103.4]
    }
  ],
  "image_size": { "w": 320, "h": 213 },
  "weights_in_use": "weights/best.pt",
  "annotated_image_url": "/static/annotated/7f0e9b....png",
  "annotated_image": "data:image/png;base64,..."  // when inline=1
}
### Known issues / limitations
* Large uploads: Files bigger than UPLOAD_MAX_MB return a JSON error (often seen as “Server returned non-JSON” in UIs that expect JSON). Lower the file size or raise UPLOAD_MAX_MB.

* Cold starts on free hosts can delay the first request while weights load.

* Low-RAM defaults: MAX_SIDE=320 trades accuracy for memory. Increase on stronger hardware.

* Missing weights: If weights/best.pt is absent, the app uses yolov8n.pt, which may detect poorly on your domain.

*Class names: Customize with weights/classes.txt (one per line) if your weights don’t embed names.
### Deployment tips (Render)
* Connect your GitHub repo → Render will build from Dockerfile on each push.

* Set environment vars (e.g., PORT=7860, UPLOAD_MAX_MB=10, MODEL_PATH=weights/best.pt).

* If the app serves HTML errors, ensure you’re on the version of app.py that returns JSON error handlers (413/404/500).

* If builds act weird: Manual Deploy → Clear build cache & deploy.

  #### License & credits

* Built with Flask and Ultralytics YOLOv8.

* Add your chosen license (e.g., MIT) to clarify reuse.

* You own your trained weights.


2.templates/ contains the HTML files for the web interface.

