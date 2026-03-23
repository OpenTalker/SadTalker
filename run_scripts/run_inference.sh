#!/usr/bin/env bash
#
# SadTalker CLI Run Script
# This script provides a convenient wrapper for running SadTalker CLI inference
#

set -e

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
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

print_usage() {
    echo -e "${BLUE}Usage: $0 --audio <audio_file> --image <image_file> [options]${NC}"
    echo ""
    echo "Required arguments:"
    echo "  --audio <file>       Path to audio file (.wav, .mp3, etc.)"
    echo "  --image <file>       Path to source image (.png, .jpg, etc.)"
    echo ""
    echo "Optional arguments:"
    echo "  --enhancer <name>    Face enhancer: gfpgan, RestoreFormer, or none (default: gfpgan)"
    echo "  --preprocess <mode>  Preprocessing: crop, resize, full, extcrop (default: crop)"
    echo "  --still              Enable still mode for full body/static images"
    echo "  --output <dir>       Output directory (default: results)"
    echo "  --help               Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --audio audio.wav --image portrait.png"
    echo "  $0 --audio audio.wav --image fullbody.png --still --preprocess full"
    echo "  $0 --audio audio.wav --image photo.jpg --enhancer gfpgan --output my_results"
}

# Get script directory and project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "${SCRIPT_DIR}/.." && pwd )"

cd "${PROJECT_ROOT}"

# Check for help flag
if [[ "$1" == "--help" ]] || [[ "$1" == "-h" ]] || [[ $# -eq 0 ]]; then
    print_usage
    exit 0
fi

# Check if virtual environment exists
VENV_DIR="${PROJECT_ROOT}/venv"

if [[ ! -d "${VENV_DIR}" ]]; then
    print_error "Virtual environment not found at ${VENV_DIR}"
    print_error "Please run the installation script first"
    exit 1
fi

# Activate virtual environment
source "${VENV_DIR}/bin/activate"

# Parse arguments
AUDIO_FILE=""
IMAGE_FILE=""
ENHANCER="gfpgan"
PREPROCESS="crop"
STILL_FLAG=""
OUTPUT_DIR=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --audio)
            AUDIO_FILE="$2"
            shift 2
            ;;
        --image)
            IMAGE_FILE="$2"
            shift 2
            ;;
        --enhancer)
            ENHANCER="$2"
            shift 2
            ;;
        --preprocess)
            PREPROCESS="$2"
            shift 2
            ;;
        --still)
            STILL_FLAG="--still"
            shift
            ;;
        --output)
            OUTPUT_DIR="--result_dir $2"
            shift 2
            ;;
        *)
            print_error "Unknown argument: $1"
            print_usage
            exit 1
            ;;
    esac
done

# Validate required arguments
if [[ -z "$AUDIO_FILE" ]]; then
    print_error "Audio file is required"
    print_usage
    exit 1
fi

if [[ -z "$IMAGE_FILE" ]]; then
    print_error "Image file is required"
    print_usage
    exit 1
fi

# Check if files exist
if [[ ! -f "$AUDIO_FILE" ]]; then
    print_error "Audio file not found: $AUDIO_FILE"
    exit 1
fi

if [[ ! -f "$IMAGE_FILE" ]]; then
    print_error "Image file not found: $IMAGE_FILE"
    exit 1
fi

print_info "Running SadTalker CLI inference..."
print_info "Audio: $AUDIO_FILE"
print_info "Image: $IMAGE_FILE"
print_info "Enhancer: $ENHANCER"
print_info "Preprocess: $PREPROCESS"
[[ -n "$STILL_FLAG" ]] && print_info "Still mode: enabled"
echo ""

# Run inference
python inference.py \
    --driven_audio "$AUDIO_FILE" \
    --source_image "$IMAGE_FILE" \
    --enhancer "$ENHANCER" \
    --preprocess "$PREPROCESS" \
    $STILL_FLAG \
    $OUTPUT_DIR

print_info ""
print_info "✓ Processing complete!"
print_info "Check the results directory for output videos."
