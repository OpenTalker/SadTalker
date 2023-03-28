# Use the official NVIDIA CUDA Docker image as the base image
FROM python:3.8.16-bullseye
ARG DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    wget \
    git \
    build-essential \
    libgl1 \
    libssl-dev \
    libffi-dev \
    libxml2-dev \
    libxslt1-dev \
    zlib1g-dev \
    libjpeg-dev \
    libpng-dev \
    unzip \
    ffmpeg

# Set the working directory
WORKDIR /app

# Clone the SadTalker repository
RUN git clone https://github.com/Winfredy/SadTalker.git

# Change the working directory to SadTalker
WORKDIR /app/SadTalker

# Install PyTorch with CUDA 11.3 support
RUN pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113

# Install dlib
RUN pip install dlib-bin

# Install GFPGAN
RUN pip install git+https://github.com/TencentARC/GFPGAN

# Install dependencies from requirements.txt
RUN pip install -r requirements.txt

# Download models using the provided script
RUN chmod +x scripts/download_models.sh && scripts/download_models.sh

ENTRYPOINT ["python3", "inference.py"]
