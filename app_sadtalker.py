import os
import sys
import gradio as gr
from src.gradio_demo import SadTalker

# Constants
POSE_STYLE_MAX = 46
FACE_RESOLUTIONS = [256, 512]
PREPROCESS_TYPES = ['crop', 'resize', 'full', 'extcrop', 'extfull']
DEFAULT_BATCH_SIZE = 2
MAX_BATCH_SIZE = 10

# Check if running in webui
try:
    import webui
    in_webui = True
except ImportError:
    in_webui = False


def sadtalker_demo(checkpoint_path='checkpoints', config_path='src/config', warpfn=None):
    """
    Create the SadTalker Gradio interface.
    
    Args:
        checkpoint_path: Path to model checkpoints
        config_path: Path to configuration files
        warpfn: Optional wrapper function for the test method
    
    Returns:
        Gradio Blocks interface
    """
    
    # Initialize SadTalker model
    try:
        sad_talker = SadTalker(checkpoint_path, config_path, lazy_load=True)
    except Exception as e:
        print(f"Error loading SadTalker model: {e}")
        return None
    
    # Initialize TTS if available
    tts_talker = None
    if sys.platform != 'win32' and not in_webui:
        try:
            from src.utils.text2speech import TTSTalker
            tts_talker = TTSTalker()
        except ImportError:
            print("TTS module not available")
    
    with gr.Blocks(analytics_enabled=False) as sadtalker_interface:
        # Header
        gr.Markdown(
            "<div align='center'>"
            "<h2>😭 SadTalker: Learning Realistic 3D Motion Coefficients for "
            "Stylized Audio-Driven Single Image Talking Face Animation (CVPR 2023)</h2>"
            "<a style='font-size:18px;color: #efefef' href='https://arxiv.org/abs/2211.12194'>Arxiv</a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; "
            "<a style='font-size:18px;color: #efefef' href='https://sadtalker.github.io'>Homepage</a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; "
            "<a style='font-size:18px;color: #efefef' href='https://github.com/Winfredy/SadTalker'>Github</a>"
            "</div>"
        )
        
        with gr.Row(equal_height=False):
            # Input Column
            with gr.Column(variant='panel'):
                # Image Input
                source_image = gr.Image(
                    label="Source Image",
                    type="filepath",
                    elem_id="sadtalker_source_image"
                )
                
                # Audio Input Section
                with gr.Tabs(elem_id="sadtalker_driven_audio"):
                    with gr.TabItem('Upload Audio'):
                        driven_audio = gr.Audio(
                            label="Input audio",
                            type="filepath"
                        )
                    
                    if tts_talker is not None:
                        with gr.TabItem('Text to Speech'):
                            input_text = gr.Textbox(
                                label="Generate audio from text",
                                lines=5,
                                placeholder="Enter text here to generate audio using Coqui.ai TTS"
                            )
                            tts_button = gr.Button(
                                'Generate Audio',
                                elem_id="sadtalker_audio_generate",
                                variant='primary'
                            )
                            tts_button.click(
                                fn=tts_talker.test,
                                inputs=[input_text],
                                outputs=[driven_audio]
                            )
            
            # Settings and Output Column
            with gr.Column(variant='panel'):
                with gr.Tabs(elem_id="sadtalker_settings"):
                    with gr.TabItem('Settings'):
                        gr.Markdown(
                            "Need help? Please visit our "
                            "[best practice page](https://github.com/OpenTalker/SadTalker/blob/main/docs/best_practice.md) "
                            "for more details"
                        )
                        
                        with gr.Column(variant='panel'):
                            # Model Settings
                            pose_style = gr.Slider(
                                minimum=0,
                                maximum=POSE_STYLE_MAX,
                                step=1,
                                label="Pose style",
                                value=0,
                                info="Select pose variation (0 = neutral)"
                            )
                            
                            size_of_image = gr.Radio(
                                choices=FACE_RESOLUTIONS,
                                value=256,
                                label='Face model resolution',
                                info="Higher resolution = better quality but slower"
                            )
                            
                            preprocess_type = gr.Radio(
                                choices=PREPROCESS_TYPES,
                                value='crop',
                                label='Preprocess mode',
                                info="How to handle input image"
                            )
                            
                            # Processing Options
                            is_still_mode = gr.Checkbox(
                                label="Still Mode",
                                info="Reduces head motion, works best with 'full' preprocess"
                            )
                            
                            batch_size = gr.Slider(
                                label="Batch size",
                                minimum=1,
                                maximum=MAX_BATCH_SIZE,
                                step=1,
                                value=DEFAULT_BATCH_SIZE,
                                info="Higher batch size = faster processing but more memory"
                            )
                            
                            enhancer = gr.Checkbox(
                                label="Enable GFPGAN face enhancement",
                                info="Improves face quality but increases processing time"
                            )
                            
                            # Generate Button
                            submit = gr.Button(
                                'Generate Video',
                                elem_id="sadtalker_generate",
                                variant='primary'
                            )
                
                # Output Section
                with gr.Tabs(elem_id="sadtalker_output"):
                    with gr.TabItem('Generated Video'):
                        gen_video = gr.Video(
                            label="Output",
                            format="mp4",
                            autoplay=True
                        )
        
        # Processing Status
        with gr.Row():
            status_text = gr.Textbox(
                label="Status",
                interactive=False,
                visible=False
            )
        
        # Set up the generation pipeline
        process_fn = warpfn(sad_talker.test) if warpfn else sad_talker.test
        
        submit.click(
            fn=process_fn,
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
            outputs=[gen_video]
        )
    
    return sadtalker_interface


if __name__ == "__main__":
    demo = sadtalker_demo()
    
    if demo is not None:
        demo.queue()
        demo.launch(
            share=False,
            debug=True
        )
    else:
        print("Failed to initialize SadTalker demo")