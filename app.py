import os, sys
import tempfile
import gradio as gr
from src.gradio_demo import SadTalker  

def get_source_image(image):   
        return image

def sadtalker_demo():

    sad_talker = SadTalker(lazy_load=True)

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
                            source_image = gr.Image(label="Source image", source="upload", type="filepath").style(height=256,width=256)
 
                with gr.Tabs(elem_id="sadtalker_driven_audio"):
                    with gr.TabItem('Upload OR TTS'):
                        with gr.Column(variant='panel'):
                            driven_audio = gr.Audio(label="Input audio", source="upload", type="filepath")

                        if sys.platform != 'win32': 
                            from src.utils.text2speech import TTSTalker
                            tts_talker = TTSTalker()
                            with gr.Column(variant='panel'):
                                input_text = gr.Textbox(label="Generating audio from text", lines=5, placeholder="please enter some text here, we genreate the audio from text using @Coqui.ai TTS.")
                                tts = gr.Button('Generate audio',elem_id="sadtalker_audio_generate", variant='primary')
                                tts.click(fn=tts_talker.test, inputs=[input_text], outputs=[driven_audio])
                            

            with gr.Column(variant='panel'): 
                with gr.Tabs(elem_id="sadtalker_checkbox"):
                    with gr.TabItem('Settings'):
                        with gr.Column(variant='panel'):
                            preprocess_type = gr.Radio(['crop','resize','full'], value='crop', label='preprocess', info="How to handle input image?")
                            is_still_mode = gr.Checkbox(label="w/ Still Mode (fewer hand motion, works with preprocess `full`)")
                            enhancer = gr.Checkbox(label="w/ GFPGAN as Face enhancer")
                            submit = gr.Button('Generate', elem_id="sadtalker_generate", variant='primary')

                with gr.Tabs(elem_id="sadtalker_genearted"):
                        gen_video = gr.Video(label="Generated video", format="mp4").style(width=256)



        with gr.Row():
            examples = [
                [
                    'examples/source_image/full_body_1.png',
                    'examples/driven_audio/bus_chinese.wav',
                    'crop',
                    True,
                    False
                ],
                [
                    'examples/source_image/full_body_2.png',
                    'examples/driven_audio/japanese.wav',
                    'crop',
                    False,
                    False
                ],
                [
                    'examples/source_image/full3.png',
                    'examples/driven_audio/deyu.wav',
                    'crop',
                    False,
                    True
                ],
                [
                    'examples/source_image/full4.jpeg',
                    'examples/driven_audio/eluosi.wav',
                    'full',
                    False,
                    True
                ],
                [
                    'examples/source_image/full4.jpeg',
                    'examples/driven_audio/imagine.wav',
                    'full',
                    True,
                    True
                ],
                [
                    'examples/source_image/full_body_1.png',
                    'examples/driven_audio/bus_chinese.wav',
                    'full',
                    True,
                    False
                ],
                [
                    'examples/source_image/art_13.png',
                    'examples/driven_audio/fayu.wav',
                    'resize',
                    True,
                    False
                ],
                [
                    'examples/source_image/art_5.png',
                    'examples/driven_audio/chinese_news.wav',
                    'resize',
                    False,
                    False
                ],
                [
                    'examples/source_image/art_5.png',
                    'examples/driven_audio/RD_Radio31_000.wav',
                    'resize',
                    True,
                    True
                ],
            ]
            gr.Examples(examples=examples,
                        inputs=[
                            source_image,
                            driven_audio,
                            preprocess_type,
                            is_still_mode,
                            enhancer], 
                        outputs=[gen_video],
                        fn=sad_talker.test,
                        cache_examples=os.getenv('SYSTEM') == 'spaces') # 

        submit.click(
                    fn=sad_talker.test, 
                    inputs=[source_image,
                            driven_audio,
                            preprocess_type,
                            is_still_mode,
                            enhancer], 
                    outputs=[gen_video]
                    )

    return sadtalker_interface
 

if __name__ == "__main__":

    demo = sadtalker_demo()
    demo.launch(share=True)


