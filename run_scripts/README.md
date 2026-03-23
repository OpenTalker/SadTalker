# SadTalker Run Scripts

This directory contains convenient scripts for running SadTalker on different platforms.

## Available Scripts

### Web UI Scripts

These scripts start the Gradio-based web interface:

#### Linux
```bash
bash run_scripts/run_linux.sh
```

#### macOS
```bash
bash run_scripts/run_macos.sh
```

#### Windows
```cmd
run_scripts\run_windows.bat
```

#### Docker
```bash
cd installation/docker
docker-compose up -d
# Access at http://localhost:7860
```

### CLI Script

For command-line inference without the web interface:

```bash
bash run_scripts/run_inference.sh --audio <audio.wav> --image <image.png> [options]
```

**Options:**
- `--audio <file>` - Input audio file (required)
- `--image <file>` - Input image file (required)
- `--enhancer <name>` - Face enhancer: gfpgan, RestoreFormer, or none
- `--preprocess <mode>` - Preprocessing: crop, resize, full, extcrop
- `--still` - Enable still mode for full body/static images
- `--output <dir>` - Output directory

**Examples:**
```bash
# Basic usage
bash run_scripts/run_inference.sh \
    --audio examples/driven_audio/bus_chinese.wav \
    --image examples/source_image/full_body_1.png

# Full body with enhancement
bash run_scripts/run_inference.sh \
    --audio audio.wav \
    --image fullbody.png \
    --still \
    --preprocess full \
    --enhancer gfpgan

# Custom output directory
bash run_scripts/run_inference.sh \
    --audio audio.wav \
    --image portrait.png \
    --output my_results
```

## Prerequisites

Before running these scripts:

1. **Install SadTalker** using the appropriate installation script:
   - Linux: `bash installation/linux/install.sh`
   - macOS: `bash installation/macos/install.sh`
   - Windows: `installation\windows\install.bat`
   - Docker: See `installation/docker/README.md`

2. **Download models** (if not done during installation):
   ```bash
   bash scripts/download_models.sh
   ```

## Web UI Features

When using the web interface, you can:
- Upload source images and audio files
- Adjust processing parameters via sliders
- Preview results in real-time
- Download generated videos
- Use text-to-speech (if TTS is installed)

Access the web UI at: **http://localhost:7860**

## Troubleshooting

### Virtual Environment Not Found
```bash
# Run the installation script for your platform
bash installation/linux/install.sh  # or macos/install.sh
```

### Models Not Found
```bash
# Download models
bash scripts/download_models.sh
```

### Port Already in Use
If port 7860 is already in use, you can specify a different port:

**Linux/macOS:**
```bash
python app_sadtalker.py --server_port 8080
```

**Windows:**
```cmd
python app_sadtalker.py --server_port 8080
```

### Permission Denied (Linux/macOS)
```bash
# Make scripts executable
chmod +x run_scripts/*.sh
```

## Advanced Usage

### Custom Python Arguments

You can pass additional arguments to the Python scripts:

```bash
# Linux/macOS
bash run_scripts/run_linux.sh --server_port 8080 --share

# Windows
run_scripts\run_windows.bat --server_port 8080 --share
```

### Running in Background (Linux/macOS)

```bash
nohup bash run_scripts/run_linux.sh > sadtalker.log 2>&1 &
# Check log: tail -f sadtalker.log
```

### Direct Python Execution

If you prefer to run Python directly:

```bash
# Activate environment first
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate.bat  # Windows

# Run application
python app_sadtalker.py

# Or CLI
python inference.py --driven_audio audio.wav --source_image image.png
```
