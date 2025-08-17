# YOLO Model Deployment 

This repository contains a **YOLOv8 object detection model** deployed with **Flask** and packaged with **Docker** for easy containerized deployment.

---

##  Project Overview
- Train and serve a YOLOv8 model for object detection.  
- Expose predictions through a simple Flask web interface.  
- Run locally or inside Docker containers for portability.  

---

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
1. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # On Windows: .venv\Scripts\activate

2. Install dependencies:
pip install -r requirements.txt

3.Run the Flask app:
python app.py
The app will be available at http://localhost:7860

### With Docker:

1.Build the Docker image:
docker build -t yolo-flask .

2.Run the container:
docker run --rm -p 7860:7860 yolo-flask
Now visit http://localhost:7860

Notes

1.weights/ contains your trained YOLOv8 model file.

2.templates/ contains the HTML files for the web interface.

