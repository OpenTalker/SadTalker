#!/usr/bin/env bash
#
# SadTalker Run Script for macOS
# This script starts the SadTalker WebUI
#

set -e

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

print_info() {
    echo -e "${GREEN}$1${NC}"
}

print_warning() {
    echo -e "${YELLOW}WARNING: $1${NC}"
}

print_error() {
    echo -e "${RED}ERROR: $1${NC}"
}

# Get script directory and project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "${SCRIPT_DIR}/.." && pwd )"

cd "${PROJECT_ROOT}"

print_info "Starting SadTalker WebUI on macOS..."
print_info "Project root: ${PROJECT_ROOT}"

# Check if virtual environment exists
VENV_DIR="${PROJECT_ROOT}/venv"

if [[ ! -d "${VENV_DIR}" ]]; then
    print_error "Virtual environment not found at ${VENV_DIR}"
    print_error "Please run the installation script first:"
    print_error "  bash installation/macos/install.sh"
    exit 1
fi

# Activate virtual environment
source "${VENV_DIR}/bin/activate"
print_info "✓ Virtual environment activated"

# Check if models are downloaded
if [[ ! -d "${PROJECT_ROOT}/checkpoints" ]] || [[ -z "$(ls -A ${PROJECT_ROOT}/checkpoints)" ]]; then
    print_warning "Models not found in checkpoints directory"
    print_warning "Please download models first:"
    print_warning "  bash scripts/download_models.sh"
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Disable analytics
export GRADIO_ANALYTICS_ENABLED=False

# Check if running on Apple Silicon
ARCH=$(uname -m)
if [[ "$ARCH" == "arm64" ]]; then
    print_info "Running on Apple Silicon (M1/M2/M3)"
    print_info "Note: Performance may vary compared to NVIDIA GPU systems"
fi

print_info "Launching SadTalker WebUI..."
print_info "The WebUI will be available at: http://localhost:7860"
print_info ""

# Run the application
python app_sadtalker.py "$@"
