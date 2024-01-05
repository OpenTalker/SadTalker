import os, sys
import gradio as gr
from src.gradio_demo import SadTalker  


try:
    import webui  # in webui
    in_webui = True
except:
    in_webui = False


def toggle_audio_file(choice):
    if choice == False:
        return gr.update(visible=True), gr.update(visible=False)
    else:
        return gr.update(visible=False), gr.update(visible=True)
    
def ref_video_fn(path_of_ref_video):
    if path_of_ref_video is not None:
        return gr.update(value=True)
    else:
        return gr.update(value=False)

def sadtalker_demo(checkpoint_path='checkpoints', config_path='src/config', warpfn=None):

    sad_talker = SadTalker(checkpoint_path, config_path, lazy_load=True)

    with gr.Blocks(analytics_enabled=False) as sadtalker_interface:
        gr.Markdown("<div align='center'> <h2> 😭 SadTalker: Learning Realistic 3D Motion Coefficients for Stylized Audio-Driven Single Image Talking Face Animation (CVPR 2023) </span> </h2> \
                    <a style='font-size:18px;color: #efefef' href='https://arxiv.org/abs/2211.12194'>Arxiv</a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \
                    <a style='font-size:18px;color: #efefef' href='https://sadtalker.github.io'>Homepage</a>  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \
                     <a style='font-size:18px;color: #efefef' href='https://github.com/Winfredy/SadTalker'> Github </div>")
        
        with gr.Row(equal_height=False):
            with gr.Column(variant='panel'):
                with gr.Tabs(elem_id="sadtalker_source_image"):
                    with gr.TabItem('Upload image'):
                        with gr.Row():
                            source_image = gr.Image(label="Source image", sources="upload", type="filepath", elem_id="img2img_image", width=512)

                with gr.Tabs(elem_id="sadtalker_driven_audio"):
                    with gr.TabItem('Upload OR TTS'):
                        with gr.Column(variant='panel'):
                            driven_audio = gr.Audio(label="Input audio", sources="upload", type="filepath")

                        if sys.platform != 'win32' and not in_webui: 
                            from src.utils.text2speech import TTSTalker
                            tts_talker = TTSTalker()
                            with gr.Column(variant='panel'):
                                input_text = gr.Textbox(label="Generating audio from text", lines=5, placeholder="please enter some text here, we genreate the audio from text using @Coqui.ai TTS.")
                                tts = gr.Button('Generate audio',elem_id="sadtalker_audio_generate", variant='primary')
                                tts.click(fn=tts_talker.test, inputs=[input_text], outputs=[driven_audio])
                            
            with gr.Column(variant='panel'): 
                with gr.Tabs(elem_id="sadtalker_checkbox"):
                    with gr.TabItem('Settings'):
                        gr.Markdown("need help? please visit our [best practice page](https://github.com/OpenTalker/SadTalker/blob/main/docs/best_practice.md) for more detials")
                        with gr.Column(variant='panel'):
                            # width = gr.Slider(minimum=64, elem_id="img2img_width", maximum=2048, step=8, label="Manually Crop Width", value=512) # img2img_width
                            # height = gr.Slider(minimum=64, elem_id="img2img_height", maximum=2048, step=8, label="Manually Crop Height", value=512) # img2img_width
                            pose_style = gr.Slider(minimum=0, maximum=46, step=1, label="Pose style", value=0) # 
                            size_of_image = gr.Radio([256, 512], value=256, label='face model resolution', info="use 256/512 model?") # 
                            preprocess_type = gr.Radio(['crop', 'resize','full', 'extcrop', 'extfull'], value='crop', label='preprocess', info="How to handle input image?")
                            is_still_mode = gr.Checkbox(label="Still Mode (fewer head motion, works with preprocess `full`)")
                            batch_size = gr.Slider(label="batch size in generation", step=1, maximum=10, value=2)
                            enhancer = gr.Checkbox(label="GFPGAN as Face enhancer")
                            submit = gr.Button('Generate', elem_id="sadtalker_generate", variant='primary')
                            
                with gr.Tabs(elem_id="sadtalker_genearted"):
                        gen_video = gr.Video(label="Generated video", format="mp4", width=256)

        if warpfn:
            submit.click(
                        fn=warpfn(sad_talker.test), 
                        inputs=[source_image,
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
        else:
            submit.click(
                        fn=sad_talker.test, 
                        inputs=[source_image,
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
    demo.queue()
    demo.launch()


