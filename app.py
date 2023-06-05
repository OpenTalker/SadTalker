import os, sys
import gradio as gr
from src.gradio_demo import SadTalker  


try:
    import webui  # in webui
    in_webui = True
except:
    in_webui = False

# mimetypes.init()
# mimetypes.add_type('application/javascript', '.js')

# script_path = os.path.dirname(os.path.realpath(__file__))

# def webpath(fn):
#     if fn.startswith(script_path):
#         web_path = os.path.relpath(fn, script_path).replace('\\', '/')
#     else:
#         web_path = os.path.abspath(fn)

#     return f'file={web_path}?{os.path.getmtime(fn)}'

# def javascript_html():
#     # Ensure localization is in `window` before scripts
#     # head = f'<script type="text/javascript">{localization.localization_js(opts.localization)}</script>\n'
#     head  = 'somehead'

#     script_js = os.path.join(script_path, "assets", "script.js")
#     head += f'<script type="text/javascript" src="{webpath(script_js)}"></script>\n'

#     script_js = os.path.join(script_path, "assets", "aspectRatioOverlay.js")
#     head += f'<script type="text/javascript" src="{webpath(script_js)}"></script>\n'

#     return head

# def resize_from_to_html(width, height, scale_by):
#     target_width = int(width * scale_by)
#     target_height = int(height * scale_by)

#     if not target_width or not target_height:
#         return "no image selected"

#     return f"resize: from <span class='resolution'>{width}x{height}</span> to <span class='resolution'>{target_width}x{target_height}</span>"

# def get_source_image(image):   
#         return image

# def reload_javascript():
#     js = javascript_html()

#     def template_response(*args, **kwargs):
#         res = shared.GradioTemplateResponseOriginal(*args, **kwargs)
#         res.body = res.body.replace(b'</head>', f'{js}</head>'.encode("utf8"))
#         res.init_headers()
#         return res

#     gradio.routes.templates.TemplateResponse = template_response

# if not hasattr(shared, 'GradioTemplateResponseOriginal'):
#     shared.GradioTemplateResponseOriginal = gradio.routes.templates.TemplateResponse


def sadtalker_demo(checkpoint_path='checkpoints', config_path='src/config', warpfn=None):

    sad_talker = SadTalker(checkpoint_path, config_path, lazy_load=True)

    with gr.Blocks(analytics_enabled=False) as sadtalker_interface:
        gr.Markdown("<div align='center'> <h2> 😭 SadTalker: Learning Realistic 3D Motion Coefficients for Stylized Audio-Driven Single Image Talking Face Animation (CVPR 2023) </span> </h2> \
                    <a style='font-size:18px;color: #efefef' href='https://arxiv.org/abs/2211.12194'>Arxiv</a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \
                    <a style='font-size:18px;color: #efefef' href='https://sadtalker.github.io'>Homepage</a>  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \
                     <a style='font-size:18px;color: #efefef' href='https://github.com/Winfredy/SadTalker'> Github </div>")
        
        with gr.Row().style(equal_height=False):
            with gr.Column(variant='panel'):
                with gr.Tabs(elem_id="sadtalker_source_image"):
                    with gr.TabItem('Upload image'):
                        with gr.Row():
                            source_image = gr.Image(label="Source image", source="upload", type="filepath", elem_id="img2img_image").style(width=512)

                        
                with gr.Tabs(elem_id="sadtalker_driven_audio"):
                    with gr.TabItem('Upload OR TTS'):
                        with gr.Column(variant='panel'):
                            driven_audio = gr.Audio(label="Input audio", source="upload", type="filepath")
                        
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
                        gr.Markdown("need help? please visit our [[best practice page](https://github.com/OpenTalker/SadTalker/blob/main/docs/best_practice.md)] for more detials")
                        with gr.Column(variant='panel'):
                            # width = gr.Slider(minimum=64, elem_id="img2img_width", maximum=2048, step=8, label="Manually Crop Width", value=512) # img2img_width
                            # height = gr.Slider(minimum=64, elem_id="img2img_height", maximum=2048, step=8, label="Manually Crop Height", value=512) # img2img_width
                            with gr.Row():
                                pose_style = gr.Slider(minimum=0, maximum=46, step=1, label="Pose style", value=0) #
                                exp_weight = gr.Slider(minimum=0, maximum=3, step=0.1, label="expression scale", value=1) # 

                            with gr.Row():
                                size_of_image = gr.Radio([256, 512], value=256, label='face model resolution', info="use 256/512 model?") # 
                                preprocess_type = gr.Radio(['crop', 'resize','full', 'extcrop', 'extfull'], value='crop', label='preprocess', info="How to handle input image?")
                            
                            with gr.Row():
                                is_still_mode = gr.Checkbox(label="Still Mode (fewer hand motion, works with preprocess `full`)")
                                batch_size = gr.Slider(label="batch size in generation", step=1, maximum=10, value=2)
                                enhancer = gr.Checkbox(label="GFPGAN as Face enhancer")
                                
                            submit = gr.Button('Generate', elem_id="sadtalker_generate", variant='primary')
                            

                with gr.Tabs(elem_id="sadtalker_genearted"):
                        gen_video = gr.Video(label="Generated video", format="mp4").style(width=256)

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
                            exp_weight
                            ], 
                    outputs=[gen_video]
                    )

    return sadtalker_interface
 

if __name__ == "__main__":

    demo = sadtalker_demo()
    demo.queue()
    demo.launch()


