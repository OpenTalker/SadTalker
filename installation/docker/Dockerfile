# SadTalker Dockerfile
# This Dockerfile creates a containerized environment for running SadTalker with GPU support

FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860 \
    GRADIO_ANALYTICS_ENABLED=False

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.8 \
    python3-pip \
    python3-dev \
    git \
    wget \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt req.txt requirements3d.txt ./

# Install PyTorch and dependencies
RUN pip3 install --no-cache-dir torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 \
    --extra-index-url https://download.pytorch.org/whl/cu113

# Install Python dependencies
RUN pip3 install --no-cache-dir -r req.txt

# Optional: Install TTS for gradio demo
RUN pip3 install --no-cache-dir TTS || echo "TTS installation failed, continuing without it"

# Copy the rest of the application
COPY . .

# Create directories for models and results
RUN mkdir -p checkpoints gfpgan/weights results

# Expose port for web UI
EXPOSE 7860

# Default command runs the web UI
CMD ["python3", "app_sadtalker.py"]
