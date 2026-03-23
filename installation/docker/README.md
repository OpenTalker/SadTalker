# Docker Installation Guide for SadTalker

This directory contains Docker configuration files for running SadTalker in containerized environments.

## Prerequisites

- Docker Engine 20.10 or higher
- Docker Compose 1.29 or higher
- For GPU support: NVIDIA Docker runtime (nvidia-docker2)

## Files

- `Dockerfile` - GPU-enabled Docker image
- `Dockerfile.cpu` - CPU-only Docker image
- `docker-compose.yml` - Docker Compose configuration

## Quick Start

### 1. GPU Version (Recommended)

```bash
# Navigate to the docker directory
cd installation/docker

# Build and start the container
docker-compose up -d

# View logs
docker-compose logs -f

# Stop the container
docker-compose down
```

### 2. CPU Version

```bash
# Edit docker-compose.yml and uncomment the sadtalker-cpu service
# Comment out the sadtalker GPU service

# Build and start the container
docker-compose up -d sadtalker-cpu
```

### 3. Direct Docker Build

```bash
# GPU version
docker build -t sadtalker:latest -f installation/docker/Dockerfile .

# CPU version
docker build -t sadtalker:cpu -f installation/docker/Dockerfile.cpu .

# Run the container
docker run -d -p 7860:7860 --gpus all sadtalker:latest
```

## Downloading Models

Before running SadTalker, you need to download the pre-trained models:

```bash
# Create checkpoints directory
mkdir -p checkpoints gfpgan

# Download models (run this from the project root)
bash scripts/download_models.sh
```

Alternatively, download models manually and place them in the `checkpoints/` and `gfpgan/weights/` directories.

## Accessing the Web UI

Once the container is running, access the web interface at:
- http://localhost:7860

## Volume Mounts

The Docker setup mounts the following directories:
- `./checkpoints` - Pre-trained models
- `./gfpgan` - GFPGAN models
- `./results` - Generated videos
- `./examples` - Example input files

## Environment Variables

You can customize the following environment variables:

- `GRADIO_SERVER_NAME` - Server bind address (default: 0.0.0.0)
- `GRADIO_SERVER_PORT` - Server port (default: 7860)
- `GRADIO_ANALYTICS_ENABLED` - Enable/disable analytics (default: False)

## Troubleshooting

### GPU Not Detected

1. Check NVIDIA Docker runtime installation:
   ```bash
   docker run --rm --gpus all nvidia/cuda:11.3.1-base nvidia-smi
   ```

2. Install nvidia-docker2 if needed:
   ```bash
   # Ubuntu/Debian
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
   curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
   sudo apt-get update && sudo apt-get install -y nvidia-docker2
   sudo systemctl restart docker
   ```

### Container Crashes

1. Check logs:
   ```bash
   docker-compose logs sadtalker
   ```

2. Ensure models are downloaded:
   ```bash
   ls -la checkpoints/
   ls -la gfpgan/weights/
   ```

### Port Already in Use

Change the port mapping in `docker-compose.yml`:
```yaml
ports:
  - "8080:7860"  # Use port 8080 instead
```

## Advanced Usage

### Running CLI Commands

```bash
# Enter the container
docker exec -it sadtalker bash

# Run inference
python inference.py --driven_audio examples/driven_audio/bus_chinese.wav \
                    --source_image examples/source_image/full_body_1.png \
                    --enhancer gfpgan
```

### Building with Custom Base Image

Edit the Dockerfile to use a different CUDA version:
```dockerfile
FROM nvidia/cuda:11.6.2-cudnn8-runtime-ubuntu20.04
```

## Resource Requirements

### Minimum Requirements
- CPU: 4 cores
- RAM: 8 GB
- Disk: 10 GB

### Recommended Requirements (GPU)
- GPU: NVIDIA GPU with 6 GB VRAM
- CPU: 8 cores
- RAM: 16 GB
- Disk: 20 GB
