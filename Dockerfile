# ===== Base image =====
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7860

# OS deps
RUN apt-get update && apt-get install -y --no-install-recommends \
      libgl1 libglib2.0-0 ffmpeg curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---- Install Python deps ----
# ONLY Flask/Ultralytics/Pillow in requirements.txt
COPY requirements.txt /app/
# Install torch+torchvision from CPU index, then the rest
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch torchvision && \
    pip install --no-cache-dir -r requirements.txt

# ---- Copy app code ----
COPY app.py /app/
COPY templates/ /app/templates/
# If you have static assets, uncomment next line
# COPY static/ /app/static/
COPY weights/ /app/weights/

# Healthcheck (optional)
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -fsS http://127.0.0.1:${PORT}/health || exit 1

EXPOSE ${PORT}

RUN pip install --no-cache-dir gunicorn

CMD ["gunicorn", "-k", "gthread", "-w", "1", "--threads", "4", "-b", "0.0.0.0:7860", "app:app"]
