import os
import sys
import gradio as gr
import logging
from src.gradio_demo import SadTalker

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Check if running in webui environment
try:
    import webui
    in_webui = True
except ImportError:
    in_webui = False

def sadtalker_demo(checkpoint_path='checkpoints', config_path='src/config', warpfn=None):
    """Creates a Gradio interface for SadTalker."""
    # Validate paths
    try:
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint path {checkpoint_path} does not exist.")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config path {config_path} does not exist.")
        logger.info(f"Checkpoint path: {checkpoint_path}")
        logger.info(f"Config path: {config_path}")
    except FileNotFoundError as e:
        logger.error(str(e))
        raise

    # Validate checkpoint files
    required_checkpoints = [
        'SadTalker_V0.0.2_256.safetensors',  # Updated to match logs
        'mapping_00229-model.pth.tar'
    ]
    for chk in required_checkpoints:
        chk_path = os.path.join(checkpoint_path, chk)
        if not os.path.exists(chk_path):
            logger.error(f"Missing checkpoint file: {chk_path}")
            raise FileNotFoundError(f"Missing checkpoint file: {chk_path}")

    # Initialize SadTalker
    try:
        sad_talker = SadTalker(checkpoint_path, config_path, lazy_load=True)
        logger.info("SadTalker initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize SadTalker: {str(e)}")
        raise

    with gr.Blocks() as sadtalker_interface:
        gr.Markdown(
            """
            <div align='center'>
                <h2> 😭 SadTalker: Learning Realistic 3D Motion Coefficients for Stylized Audio-Driven Single Image Talking Face Animation (CVPR 2023) </h2>
                <a style='font-size:18px;color: #efefef' href='https://arxiv.org/abs/2211.12194'>Arxiv</a>   
                <a style='font-size:18px;color: #efefef' href='https://sadtalker.github.io'>Homepage</a>   
                <a style='font-size:18px;color: #efefef' href='https://github.com/Winfredy/SadTalker'>Github</a>
            </div>
            """
        )

        with gr.Row():
            # Input Column
            with gr.Column(variant='panel'):
                gr.Markdown("### Upload Inputs")
                source_image = gr.File(
                    label="Source Image (PNG/JPG, ideally 512x512)",
                    file_types=[".png", ".jpg", ".jpeg"],
                    interactive=True
                )
                image_preview = gr.Image(label="Image Preview", interactive=False)
                image_status = gr.Textbox(label="Image Upload Status", interactive=False)
                driven_audio = gr.Audio(
                    label="Input Audio (WAV/MP3)",
                    type="filepath",
                    interactive=True
                )
                audio_status = gr.Textbox(label="Audio Upload Status", interactive=False)
                error_message = gr.Textbox(label="Error Message", interactive=False, visible=False)

                # Log upload events
                source_image.upload(
                    fn=lambda x: (f"Image uploaded: {x}", x),
                    inputs=[source_image],
                    outputs=[image_status, image_preview]
                )
                driven_audio.upload(
                    fn=lambda x: f"Audio uploaded: {x}",
                    inputs=[driven_audio],
                    outputs=[audio_status]
                )

            # Settings and Output Column
            with gr.Column(variant='panel'):
                gr.Markdown("### Settings and Output")
                pose_style = gr.Slider(
                    minimum=0,
                    maximum=46,
                    step=1,
                    label="Pose Style",
                    value=0
                )
                size_of_image = gr.Radio(
                    [256, 512],
                    value=256,
                    label="Face Model Resolution",
                    info="Use 256/512 model?"
                )
                preprocess_type = gr.Radio(
                    ['crop', 'resize', 'full', 'extcrop', 'extfull'],
                    value='crop',
                    label="Preprocess",
                    info="How to handle input image?"
                )
                is_still_mode = gr.Checkbox(
                    label="Still Mode (fewer head motions, works with preprocess 'full')"
                )
                batch_size = gr.Slider(
                    label="Batch Size in Generation",
                    step=1,
                    minimum=1,
                    maximum=10,
                    value=2
                )
                enhancer = gr.Checkbox(label="GFPGAN as Face Enhancer")
                submit = gr.Button('Generate', variant='primary')
                gen_video = gr.Video(
                    label="Generated Video (MP4)",
                    interactive=False
                )

        # Submit button logic with input validation
        def validate_and_run(source_image, driven_audio, preprocess_type, is_still_mode, enhancer, batch_size, size_of_image, pose_style):
            try:
                logger.info(f"Received inputs: image={source_image}, audio={driven_audio}")
                # Handle gr.File and gr.Audio inputs
                image_path = source_image.name if hasattr(source_image, 'name') else source_image
                audio_path = driven_audio if driven_audio else None
                logger.info(f"Processed paths: image={image_path}, audio={audio_path}")
                if not image_path or not os.path.exists(image_path):
                    logger.error("Invalid or missing source image")
                    return None, "Error: Please upload a valid source image."
                if not audio_path or not os.path.exists(audio_path):
                    logger.error("Invalid or missing audio file")
                    return None, "Error: Please upload a valid audio file."
                if not isinstance(audio_path, (str, bytes, os.PathLike)):
                    logger.error(f"Invalid audio path type: {type(audio_path)}")
                    return None, f"Error: Audio input must be a file path, not {type(audio_path)}."
                logger.info("Inputs validated successfully")
                result = (warpfn(sad_talker.test) if warpfn else sad_talker.test)(
                    image_path, audio_path, preprocess_type, is_still_mode, enhancer, batch_size, size_of_image, pose_style
                )
                logger.info("Video generation completed")
                return result, None
            except Exception as e:
                logger.error(f"Error during video generation: {str(e)}")
                return None, f"Error: {str(e)}"

        submit.click(
            fn=validate_and_run,
            inputs=[
                source_image,
                driven_audio,
                preprocess_type,
                is_still_mode,
                enhancer,
                batch_size,
                size_of_image,
                pose_style
            ],
            outputs=[gen_video, error_message]
        )

    return sadtalker_interface


if __name__ == "__main__":
    try:
        demo = sadtalker_demo()
        demo.queue()
        demo.launch()
    except Exception as e:
        logger.error(f"Failed to launch Gradio interface: {str(e)}")
        raise
