#!/usr/bin/env bash
#
# SadTalker Installation Script for Linux
# This script automates the installation process for SadTalker on Linux systems
#

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Delimiter for output
delimiter="################################################################"

# Print with color
print_color() {
    local color=$1
    shift
    echo -e "${color}$@${NC}"
}

print_header() {
    echo ""
    echo "${delimiter}"
    print_color "${GREEN}" "$1"
    echo "${delimiter}"
    echo ""
}

print_error() {
    print_color "${RED}" "ERROR: $1"
}

print_warning() {
    print_color "${YELLOW}" "WARNING: $1"
}

print_info() {
    print_color "${BLUE}" "$1"
}

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   print_error "This script should not be run as root (do not use sudo)"
   exit 1
fi

print_header "SadTalker Installation Script for Linux"

# Get script directory and project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "${SCRIPT_DIR}/../.." && pwd )"

cd "${PROJECT_ROOT}"

print_info "Project root: ${PROJECT_ROOT}"

# Step 1: Check prerequisites
print_header "Step 1: Checking prerequisites"

# Check Python
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

PYTHON_VERSION=$(python3 --version | grep -oP '\d+\.\d+')
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [[ $PYTHON_MAJOR -lt 3 ]] || [[ $PYTHON_MAJOR -eq 3 && $PYTHON_MINOR -lt 8 ]]; then
    print_error "Python 3.8 or higher is required. Found: $(python3 --version)"
    exit 1
fi

print_info "✓ Python $(python3 --version | grep -oP '\d+\.\d+\.\d+') found"

# Check git
if ! command -v git &> /dev/null; then
    print_error "Git is not installed. Please install git."
    exit 1
fi
print_info "✓ Git found"

# Check if venv module is available
if ! python3 -c "import venv" &> /dev/null; then
    print_error "Python venv module is not available. Please install python3-venv."
    print_info "On Ubuntu/Debian: sudo apt-get install python3-venv"
    print_info "On Fedora: sudo dnf install python3-venv"
    exit 1
fi
print_info "✓ Python venv module found"

# Check ffmpeg
if ! command -v ffmpeg &> /dev/null; then
    print_warning "FFmpeg is not installed. It will be installed via conda/pip."
    print_info "Alternatively, install it with your package manager:"
    print_info "  Ubuntu/Debian: sudo apt-get install ffmpeg"
    print_info "  Fedora: sudo dnf install ffmpeg"
else
    print_info "✓ FFmpeg found"
fi

# Step 2: Create virtual environment
print_header "Step 2: Creating virtual environment"

VENV_DIR="${PROJECT_ROOT}/venv"

if [[ -d "${VENV_DIR}" ]]; then
    print_warning "Virtual environment already exists at ${VENV_DIR}"
    read -p "Do you want to remove it and create a new one? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "${VENV_DIR}"
        print_info "Removed existing virtual environment"
    else
        print_info "Skipping virtual environment creation"
    fi
fi

if [[ ! -d "${VENV_DIR}" ]]; then
    python3 -m venv "${VENV_DIR}"
    print_info "✓ Virtual environment created at ${VENV_DIR}"
fi

# Activate virtual environment
source "${VENV_DIR}/bin/activate"
print_info "✓ Virtual environment activated"

# Upgrade pip
print_info "Upgrading pip..."
python -m pip install --upgrade pip

# Step 3: Install PyTorch
print_header "Step 3: Installing PyTorch"

# Detect GPU
if command -v nvidia-smi &> /dev/null; then
    print_info "NVIDIA GPU detected"
    TORCH_COMMAND="pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113"
elif lspci 2>/dev/null | grep -i vga | grep -i amd &> /dev/null; then
    print_info "AMD GPU detected"
    TORCH_COMMAND="pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/rocm5.2"
else
    print_warning "No GPU detected. Installing CPU-only version of PyTorch"
    TORCH_COMMAND="pip install torch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1"
fi

print_info "Installing PyTorch..."
eval $TORCH_COMMAND

print_info "✓ PyTorch installed"

# Step 4: Install dependencies
print_header "Step 4: Installing dependencies"

if [[ -f "${PROJECT_ROOT}/req.txt" ]]; then
    pip install -r "${PROJECT_ROOT}/req.txt"
    print_info "✓ Dependencies from req.txt installed"
else
    print_error "req.txt not found in ${PROJECT_ROOT}"
    exit 1
fi

# Install TTS (optional but recommended for gradio demo)
print_info "Installing TTS (optional, for gradio demo with text-to-speech)..."
pip install TTS || print_warning "TTS installation failed. This is optional."

# Step 5: Download models
print_header "Step 5: Downloading models"

if [[ -f "${PROJECT_ROOT}/scripts/download_models.sh" ]]; then
    print_info "Downloading pre-trained models..."
    bash "${PROJECT_ROOT}/scripts/download_models.sh"
    print_info "✓ Models downloaded"
else
    print_warning "Model download script not found. You may need to download models manually."
    print_info "Please refer to the README for model download instructions."
fi

# Step 6: Installation complete
print_header "Installation Complete!"

print_info "SadTalker has been successfully installed."
print_info ""
print_info "To activate the environment in the future, run:"
print_info "  source ${VENV_DIR}/bin/activate"
print_info ""
print_info "To run SadTalker:"
print_info "  1. WebUI mode: bash run_scripts/run_linux.sh"
print_info "  2. CLI mode: python inference.py --driven_audio <audio.wav> --source_image <image.png>"
print_info ""
print_info "For more information, see the README.md file."

echo ""
echo "${delimiter}"
