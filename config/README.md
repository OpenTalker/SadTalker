# SadTalker Configuration

This directory contains configuration files for SadTalker.

## Configuration Files

### config.ini
Main configuration file with default parameters for SadTalker processing.

## Configuration Parameters

### Model Configuration
- `checkpoint_dir`: Directory where model checkpoints are stored
- `face_model_resolution`: Resolution of face model (256 or 512)
- `mapping_model`: Mapping model version to use

### Processing Parameters
- `preprocess`: Preprocessing mode
  - `crop`: Crop face region
  - `resize`: Resize input
  - `full`: Full image processing
  - `extcrop`: Extended crop
- `still_mode`: Enable for full body/static image generation
- `ref_pose`: Optional reference video for pose
- `ref_eyeblink`: Optional reference video for eye blinks

### Enhancement Options
- `enhancer`: Face enhancement model
  - `gfpgan`: GFPGAN face enhancer
  - `RestoreFormer`: RestoreFormer enhancer
  - `none`: No enhancement
- `background_enhancer`: Background enhancement
  - `realesrgan`: Real-ESRGAN enhancer
  - `none`: No enhancement

### Output Settings
- `result_dir`: Directory for generated videos
- `pose_style`: Pose style index (0-45)
- `exp_scale`: Expression scale factor (0.0-3.0)
- `batch_size`: Batch size for processing
- `size`: Size of face region

### Advanced Options
- `use_3dmm`: Enable 3D morphable model
- `use_face_parsing`: Enable face parsing for background
- `use_idle_mode`: Enable idle mode
- `pose_fps`: FPS for pose extraction

## Usage

You can override these settings using command-line arguments when running SadTalker:

```bash
python inference.py \
    --driven_audio audio.wav \
    --source_image image.png \
    --enhancer gfpgan \
    --preprocess full \
    --still
```

## Creating Custom Configurations

Copy `config.ini` to create custom configuration files:

```bash
cp config/config.ini config/my_config.ini
# Edit my_config.ini with your preferences
```

Then use it in your scripts by referencing the configuration values.
