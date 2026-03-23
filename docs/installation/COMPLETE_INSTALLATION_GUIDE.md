# Complete Installation Guide for SadTalker

This guide provides detailed installation instructions for all supported platforms.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Installation Methods](#installation-methods)
  - [Automated Installation](#automated-installation)
  - [Manual Installation](#manual-installation)
  - [Docker Installation](#docker-installation)
- [Model Downloads](#model-downloads)
- [Troubleshooting](#troubleshooting)
- [Verification](#verification)

## Prerequisites

### All Platforms
- **Python**: 3.8 or higher
- **Git**: For cloning the repository
- **FFmpeg**: For video/audio processing
- **Disk Space**: At least 10 GB free (for models and dependencies)

### Platform-Specific Requirements

#### Linux
- Build tools: `gcc`, `g++`, `make`
- Python development headers: `python3-dev`
- Virtual environment support: `python3-venv`

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install python3 python3-pip python3-venv git ffmpeg build-essential
```

**Fedora:**
```bash
sudo dnf install python3 python3-pip python3-devel git ffmpeg gcc gcc-c++ make
```

#### macOS
- **Xcode Command Line Tools**: `xcode-select --install`
- **Homebrew** (recommended): Install from https://brew.sh

**With Homebrew:**
```bash
brew install python@3.10 git ffmpeg
```

#### Windows
- **Python 3.8+**: Download from https://www.python.org/downloads/windows/
  - ✅ Check "Add Python to PATH" during installation
- **Git**: Download from https://git-scm.com/download/win
- **FFmpeg**: Download from https://ffmpeg.org/download.html
  - Or use package managers:
    - Scoop: `scoop install ffmpeg`
    - Chocolatey: `choco install ffmpeg`

#### Docker
- **Docker Engine**: 20.10 or higher
- **Docker Compose**: 1.29 or higher
- **For GPU support**: NVIDIA Docker runtime (nvidia-docker2)

## Installation Methods

### Automated Installation

The easiest way to install SadTalker is using our automated installation scripts.

#### Linux

```bash
# Clone the repository
git clone https://github.com/OpenTalker/SadTalker.git
cd SadTalker

# Run the installation script
bash installation/linux/install.sh
```

The script will:
1. ✅ Check prerequisites
2. ✅ Create a virtual environment
3. ✅ Install PyTorch (with GPU support if available)
4. ✅ Install all dependencies
5. ✅ Download pre-trained models

#### macOS

```bash
# Clone the repository
git clone https://github.com/OpenTalker/SadTalker.git
cd SadTalker

# Run the installation script
bash installation/macos/install.sh
```

**Note for Apple Silicon (M1/M2/M3)**: The script will install PyTorch with MPS support, which may have limitations compared to NVIDIA GPUs.

#### Windows

1. Clone the repository:
```cmd
git clone https://github.com/OpenTalker/SadTalker.git
cd SadTalker
```

2. Run the installation script:
```cmd
installation\windows\install.bat
```

3. Follow the on-screen prompts

**Note**: Model download on Windows may require manual steps. See [Model Downloads](#model-downloads).

### Manual Installation

If you prefer to install manually or need more control:

#### Step 1: Clone and Setup Virtual Environment

```bash
# Clone repository
git clone https://github.com/OpenTalker/SadTalker.git
cd SadTalker

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# Linux/macOS:
source venv/bin/activate
# Windows:
venv\Scripts\activate.bat
```

#### Step 2: Install PyTorch

Choose the appropriate command for your system:

**NVIDIA GPU (CUDA 11.3):**
```bash
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
```

**AMD GPU (ROCm 5.2):**
```bash
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/rocm5.2
```

**CPU Only:**
```bash
pip install torch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1
```

**macOS (including Apple Silicon):**
```bash
pip install torch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1
```

#### Step 3: Install Dependencies

```bash
# Linux/macOS
pip install -r req.txt

# Windows
pip install -r requirements.txt
```

#### Step 4: Install Optional TTS

```bash
pip install TTS
```

**Note**: TTS may not work on all platforms and is optional.

#### Step 5: Download Models

See [Model Downloads](#model-downloads) section below.

### Docker Installation

Docker provides the easiest way to run SadTalker with all dependencies pre-configured.

#### GPU Version (Recommended)

1. **Install NVIDIA Docker** (if you haven't already):
```bash
# Ubuntu/Debian
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

2. **Build and Run**:
```bash
git clone https://github.com/OpenTalker/SadTalker.git
cd SadTalker/installation/docker
docker-compose up -d
```

3. **Access the Web UI**: http://localhost:7860

For detailed Docker instructions, see [installation/docker/README.md](../installation/docker/README.md).

#### CPU Version

Edit `docker-compose.yml` to use the CPU service instead, then:
```bash
docker-compose up -d sadtalker-cpu
```

## Model Downloads

### Automatic Download (Linux/macOS)

```bash
bash scripts/download_models.sh
```

This will download all required models to the `checkpoints/` and `gfpgan/weights/` directories.

### Manual Download

If automatic download fails, you can download models manually:

#### Pre-trained Models
Download from one of these sources:
- [Google Drive](https://drive.google.com/file/d/1gwWh45pF7aelNP_P78uDJL8Sycep-K7j/view?usp=sharing)
- [GitHub Releases](https://github.com/OpenTalker/SadTalker/releases)
- [Baidu (百度云盘)](https://pan.baidu.com/s/1kb1BCPaLOWX1JJb9Czbn6w?pwd=sadt) (Password: `sadt`)

#### GFPGAN Models
Download from:
- [Google Drive](https://drive.google.com/file/d/19AIBsmfcHW6BRJmeqSFlG5fL445Xmsyi?usp=sharing)
- [GitHub Releases](https://github.com/OpenTalker/SadTalker/releases)

#### Installation Steps
1. Download the model archives
2. Extract to the SadTalker directory:
   - Main models → `checkpoints/`
   - GFPGAN models → `gfpgan/weights/`

Expected directory structure:
```
SadTalker/
├── checkpoints/
│   ├── mapping_00229-model.pth.tar
│   ├── mapping_00109-model.pth.tar
│   ├── SadTalker_V0.0.2_256.safetensors
│   └── SadTalker_V0.0.2_512.safetensors
└── gfpgan/
    └── weights/
        ├── alignment_WFLW_4HG.pth
        ├── detection_Resnet50_Final.pth
        ├── GFPGANv1.4.pth
        └── parsing_parsenet.pth
```

## Troubleshooting

### Common Issues

#### "Python not found" or "command not found"
- **Solution**: Ensure Python is installed and in your PATH
- **Windows**: Reinstall Python and check "Add Python to PATH"
- **Linux/macOS**: Install Python using your package manager

#### "FFmpeg not found"
- **Solution**: Install FFmpeg using your package manager
- **Linux**: `sudo apt-get install ffmpeg` or `sudo dnf install ffmpeg`
- **macOS**: `brew install ffmpeg`
- **Windows**: Download from ffmpeg.org or use `scoop install ffmpeg`

#### "No module named 'torch'"
- **Solution**: PyTorch installation failed. Try:
```bash
pip install torch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1
```

#### "CUDA out of memory"
- **Solution**: Your GPU doesn't have enough VRAM
  - Try using smaller batch size
  - Use 256 model instead of 512
  - Process on CPU instead

#### Virtual environment activation fails
- **Linux/macOS**: Ensure you have execute permissions: `chmod +x venv/bin/activate`
- **Windows**: Use `venv\Scripts\activate.bat` not `activate.sh`

#### Models not downloading
- **Solution**: Download manually (see [Manual Download](#manual-download))
- Check your internet connection
- Try a different download source

#### "Permission denied" errors (Linux/macOS)
```bash
# Make scripts executable
chmod +x installation/linux/install.sh
chmod +x run_scripts/*.sh
```

### Platform-Specific Issues

#### macOS Apple Silicon (M1/M2/M3)
- Some operations may be slower due to PyTorch MPS limitations
- TTS may not work reliably
- Use CPU mode if MPS causes issues

#### Windows
- Ensure all prerequisites are in PATH
- Some packages may require Microsoft C++ Build Tools
- Use Git Bash for running .sh scripts if needed

#### Docker
- Ensure Docker daemon is running: `sudo systemctl start docker`
- For GPU: Verify nvidia-docker: `docker run --rm --gpus all nvidia/cuda:11.3.1-base nvidia-smi`
- Check logs: `docker-compose logs sadtalker`

## Verification

### Verify Installation

After installation, verify everything is working:

#### 1. Check Python and Packages
```bash
# Activate environment
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate.bat  # Windows

# Check installations
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import cv2; print('OpenCV installed')"
python -c "import gradio; print('Gradio installed')"
```

#### 2. Check Models
```bash
ls checkpoints/
ls gfpgan/weights/
```

You should see the model files listed above.

#### 3. Run a Test
```bash
# Start the Web UI
bash run_scripts/run_linux.sh  # or run_macos.sh or run_windows.bat

# Or run CLI inference
python inference.py \
    --driven_audio examples/driven_audio/bus_chinese.wav \
    --source_image examples/source_image/full_body_1.png \
    --enhancer gfpgan
```

### Expected Output

If everything is working:
- ✅ Web UI opens at http://localhost:7860
- ✅ CLI generates video in `results/` directory
- ✅ No error messages in console

## Next Steps

After successful installation:

1. **Run SadTalker**: See [run_scripts/README.md](../run_scripts/README.md)
2. **Configure settings**: See [config/README.md](../config/README.md)
3. **Read best practices**: See [best_practice.md](best_practice.md)
4. **Check FAQ**: See [FAQ.md](FAQ.md)

## Getting Help

If you encounter issues:

1. Check [FAQ.md](FAQ.md)
2. Search [existing issues](https://github.com/OpenTalker/SadTalker/issues)
3. Open a new issue with:
   - Your platform and OS version
   - Python version
   - Error messages
   - Steps to reproduce

## Contributing

Found a bug in the installation process or have suggestions? Please open an issue or submit a pull request!
