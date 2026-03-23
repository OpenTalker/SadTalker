# SadTalker Project Structure

This document describes the organization and structure of the SadTalker project.

## Directory Structure

```
SadTalker/
├── installation/           # Installation scripts for all platforms
│   ├── linux/             # Linux installation
│   │   └── install.sh     # Automated Linux installation script
│   ├── macos/             # macOS installation
│   │   └── install.sh     # Automated macOS installation script
│   ├── windows/           # Windows installation
│   │   └── install.bat    # Automated Windows installation script
│   └── docker/            # Docker configuration
│       ├── Dockerfile     # GPU-enabled Docker image
│       ├── Dockerfile.cpu # CPU-only Docker image
│       ├── docker-compose.yml
│       └── README.md      # Docker installation guide
│
├── run_scripts/           # Convenient scripts to run SadTalker
│   ├── run_linux.sh       # Run WebUI on Linux
│   ├── run_macos.sh       # Run WebUI on macOS
│   ├── run_windows.bat    # Run WebUI on Windows
│   ├── run_inference.sh   # CLI inference wrapper script
│   └── README.md          # Run scripts documentation
│
├── config/                # Configuration files
│   ├── config.ini         # Default configuration
│   └── README.md          # Configuration guide
│
├── src/                   # Source code
│   ├── face3d/           # 3D face reconstruction
│   ├── facerender/       # Face rendering
│   ├── audio2pose_models/
│   ├── audio2exp_models/
│   └── utils/
│
├── scripts/               # Utility scripts
│   ├── download_models.sh # Model download script
│   ├── extension.py       # SD WebUI extension
│   └── test.sh
│
├── docs/                  # Documentation
│   ├── install.md         # Additional installation guides
│   ├── best_practice.md   # Best practices
│   ├── FAQ.md             # Frequently asked questions
│   ├── webui_extension.md # WebUI extension docs
│   └── changlelog.md      # Change log
│
├── examples/              # Example files
│   ├── source_image/      # Example source images
│   └── driven_audio/      # Example audio files
│
├── checkpoints/           # Pre-trained models (downloaded separately)
│   ├── mapping_*.pth.tar
│   └── SadTalker_*.safetensors
│
├── gfpgan/               # GFPGAN models for face enhancement
│   └── weights/
│
├── results/              # Generated output videos
│
├── venv/                 # Python virtual environment (created during install)
│
├── app_sadtalker.py      # Gradio web UI application
├── inference.py          # CLI inference script
├── launcher.py           # Legacy launcher
├── predict.py            # Prediction script
├── requirements.txt      # Python dependencies (Windows)
├── req.txt               # Python dependencies (Linux/macOS)
├── requirements3d.txt    # 3D-specific dependencies
├── webui.sh              # Legacy WebUI launcher (Linux/macOS)
├── webui.bat             # Legacy WebUI launcher (Windows)
├── LICENSE               # Apache 2.0 license
└── README.md             # This file
```

## Quick Links

### Installation
- **Linux**: [`installation/linux/install.sh`](installation/linux/install.sh)
- **macOS**: [`installation/macos/install.sh`](installation/macos/install.sh)
- **Windows**: [`installation/windows/install.bat`](installation/windows/install.bat)
- **Docker**: [`installation/docker/README.md`](installation/docker/README.md)

### Running
- **Run Scripts**: [`run_scripts/README.md`](run_scripts/README.md)
- **Web UI**: `bash run_scripts/run_linux.sh` (or `run_macos.sh`, `run_windows.bat`)
- **CLI**: `bash run_scripts/run_inference.sh --audio <file> --image <file>`

### Configuration
- **Config Files**: [`config/README.md`](config/README.md)
- **Default Config**: [`config/config.ini`](config/config.ini)

### Documentation
- **Best Practices**: [`docs/best_practice.md`](docs/best_practice.md)
- **FAQ**: [`docs/FAQ.md`](docs/FAQ.md)
- **Installation Guide**: [`docs/install.md`](docs/install.md)

## Key Files

### Core Application Files
- `app_sadtalker.py` - Main Gradio web interface
- `inference.py` - Command-line inference script
- `launcher.py` - Automatic dependency installation and launcher

### Configuration Files
- `requirements.txt` - Python dependencies for Windows
- `req.txt` - Python dependencies for Linux/macOS
- `requirements3d.txt` - Optional 3D reconstruction dependencies
- `config/config.ini` - Default configuration parameters

### Legacy Files
- `webui.sh` - Legacy WebUI launcher (superseded by `run_scripts/run_linux.sh`)
- `webui.bat` - Legacy Windows launcher (superseded by `run_scripts/run_windows.bat`)

## Usage Workflows

### First-Time Setup
1. Run installation script for your platform
2. Models will be downloaded automatically (or manually if needed)
3. Virtual environment will be created and configured

### Running the Application
1. Use the appropriate run script for your platform
2. Web UI will be available at http://localhost:7860
3. Or use CLI for batch processing

### Development
1. Activate virtual environment: `source venv/bin/activate`
2. Make changes to source code in `src/`
3. Test changes using run scripts or direct Python execution

## Migration from Old Structure

If you were using the old installation method:

**Old Way:**
```bash
bash webui.sh  # or webui.bat
```

**New Way:**
```bash
bash run_scripts/run_linux.sh  # or run_macos.sh or run_windows.bat
```

The new structure provides:
- ✅ Cleaner organization with dedicated folders
- ✅ Platform-specific installation scripts
- ✅ Docker support out of the box
- ✅ Centralized configuration
- ✅ Better documentation
- ✅ Easier maintenance and updates

All existing functionality remains the same, just better organized!
