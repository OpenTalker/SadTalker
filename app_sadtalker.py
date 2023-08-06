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

        if not in_webui:
            gr.Markdown("<div align='center'> <h2> 😭 SadTalker: Learning Realistic 3D Motion Coefficients for Stylized Audio-Driven Single Image Talking Face Animation (CVPR 2023) </span> </h2> \
                        <a style='font-size:18px;color: #efefef' href='https://arxiv.org/abs/2211.12194'>Arxiv</a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \
                        <a style='font-size:18px;color: #efefef' href='https://sadtalker.github.io'>Homepage</a>  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \
                        <a style='font-size:18px;color: #efefef' href='https://github.com/Winfredy/SadTalker'> Github </div>")
        
        with gr.Row().style(equal_height=False):
            with gr.Column(variant='panel'):
                with gr.Tabs(elem_id="sadtalker_source_image"):
                    with gr.TabItem('Source image'):
                        with gr.Row():
                            source_image = gr.Image(label="Source image", source="upload", type="filepath", elem_id="img2img_image").style(width=512)


                with gr.Tabs(elem_id="sadtalker_driven_audio"):
                    with gr.TabItem('Driving Methods'):
                        gr.Markdown("Possible driving combinations: <br> 1. Audio only 2. Audio/IDLE Mode + Ref Video(pose, blink, pose+blink) 3. IDLE Mode only 4. Ref Video only (all) ")

                        with gr.Row():
                            driven_audio = gr.Audio(label="Input audio", source="upload", type="filepath")
                            driven_audio_no = gr.Audio(label="Use IDLE mode, no audio is required", source="upload", type="filepath", visible=False)

                            with gr.Column():
                                use_idle_mode = gr.Checkbox(label="Use Idle Animation")
                                length_of_audio = gr.Number(value=5, label="The length(seconds) of the generated video.")
                                use_idle_mode.change(toggle_audio_file, inputs=use_idle_mode, outputs=[driven_audio, driven_audio_no]) # todo

                                if sys.platform != 'win32' and not in_webui:
                                    with gr.Accordion('Generate Audio From TTS', open=False):
                                        from src.utils.text2speech import TTSTalker
                                        tts_talker = TTSTalker()
                                        with gr.Column(variant='panel'):
                                            input_text = gr.Textbox(label="Generating audio from text", lines=5, placeholder="please enter some text here, we genreate the audio from text using @Coqui.ai TTS.")
                                            tts = gr.Button('Generate audio',elem_id="sadtalker_audio_generate", variant='primary')
                                            tts.click(fn=tts_talker.test, inputs=[input_text], outputs=[driven_audio])

                        with gr.Row():
                            ref_video = gr.Video(label="Reference Video", source="upload", type="filepath", elem_id="vidref").style(width=512)

                            with gr.Column():
                                use_ref_video = gr.Checkbox(label="Use Reference Video")
                                ref_info = gr.Radio(['pose', 'blink','pose+blink', 'all'], value='pose', label='Reference Video',info="How to borrow from reference Video?((fully transfer, aka, video driving mode))")

                            ref_video.change(ref_video_fn, inputs=ref_video, outputs=[use_ref_video]) # todo


            with gr.Column(variant='panel'): 
                with gr.Tabs(elem_id="sadtalker_checkbox"):
                    with gr.TabItem('Settings'):
                        gr.Markdown("need help? please visit our [[best practice page](https://github.com/OpenTalker/SadTalker/blob/main/docs/best_practice.md)] for more detials")
                        with gr.Column(variant='panel'):
                            # width = gr.Slider(minimum=64, elem_id="img2img_width", maximum=2048, step=8, label="Manually Crop Width", value=512) # img2img_width
                            # height = gr.Slider(minimum=64, elem_id="img2img_height", maximum=2048, step=8, label="Manually Crop Height", value=512) # img2img_width
                            with gr.Row():
                                pose_style = gr.Slider(minimum=0, maximum=45, step=1, label="Pose style", value=0) #
                                exp_weight = gr.Slider(minimum=0, maximum=3, step=0.1, label="expression scale", value=1) # 
                                blink_every = gr.Checkbox(label="use eye blink", value=True)

                            with gr.Row():
                                size_of_image = gr.Radio([256, 512], value=256, label='face model resolution', info="use 256/512 model?") # 
                                preprocess_type = gr.Radio(['crop', 'resize','full', 'extcrop', 'extfull'], value='crop', label='preprocess', info="How to handle input image?")
                            
                            with gr.Row():
                                is_still_mode = gr.Checkbox(label="Still Mode (fewer head motion, works with preprocess `full`)")
                                facerender = gr.Radio(['facevid2vid','pirender'], value='facevid2vid', label='facerender', info="which face render?")
                                
                            with gr.Row():
                                batch_size = gr.Slider(label="batch size in generation", step=1, maximum=10, value=1)
                                enhancer = gr.Checkbox(label="GFPGAN as Face enhancer")
                            
                            submit = gr.Button('Generate', elem_id="sadtalker_generate", variant='primary')
                            
                with gr.Tabs(elem_id="sadtalker_genearted"):
                        gen_video = gr.Video(label="Generated video", format="mp4").style(width=256)

        

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
                            pose_style,
                            facerender,
                            exp_weight,
                            use_ref_video,
                            ref_video,
                            ref_info,
                            use_idle_mode,
                            length_of_audio,
                            blink_every
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
                            pose_style,
                            facerender,
                            exp_weight,
                            use_ref_video,
                            ref_video,
                            ref_info,
                            use_idle_mode,
                            length_of_audio,
                            blink_every
                            ], 
                    outputs=[gen_video]
                    )

    return sadtalker_interface
 

if __name__ == "__main__":

    demo = sadtalker_demo()
    demo.queue()
    demo.launch()


