import gradio as gr

tts = None
import torch
from TTS.api import TTS

device = "cuda" if torch.cuda.is_available() else "cpu"
# tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)
# available_models = TTS.list_models(None)
available_speakers = tts.speakers
available_languages = tts.languages

def gradio_error_wrap(fun):
    def inner_fun(*args, **kwargs):
        try:
            return fun(*args, **kwargs)
        except Exception as e:
            raise gr.Error(e)
    return inner_fun

@gradio_error_wrap
def tts_predict(
    input_text: str,
    language: str,
    use_custom:bool, ref_select:str, ref: str,
    clean_ref: bool,
    auto_det_lang: bool,
    tos: bool,
):
    global tts
    if not tos:
        gr.Warning(
            "Please accept the Terms & Condition from coqui tts, which is at https://coqui.ai/cpml . "
        )
        return None

    output_filepath = ""
    import tempfile

    output_filepath = tempfile.NamedTemporaryFile(
        prefix="audio", suffix=".wav"
    ).name  # 操作系统会在python结束的时候close这个文件，然后删除它。
    print("准备输出到", output_filepath)
    
    if use_custom:
        tts.tts_to_file(
            text=input_text, speaker_wav=ref, 
            language=language, file_path=output_filepath
        )
    else:
        tts.tts_to_file(
            text=input_text, speaker=ref_select, 
            language=language, file_path=output_filepath
        )
    return output_filepath


def make_tts_ui(output_audio_gr):
    input_text_gr = gr.Textbox(
        label="Text Prompt",
        info="One or two sentences at a time is better. Up to 200 text characters.",
        value="Hi there, I'm your new voice clone. Try your best to upload quality audio.",
    )
    language_gr = gr.Dropdown(
        label="Language",
        info="Select an output language for the synthesised speech",
        choices=available_languages,
        max_choices=1,
        value="en",
    )
    
    use_custom_ref = gr.Checkbox(
        label="Use customized reference audio as speaker",
        value=False,
        info="If enabled, we will use the uploaded audio file instead of the selected speaker name.",
    )
    ref_select_gr = gr.Dropdown(
        label="Speaker",
        info="Select an speaker supported by the model",
        choices=available_speakers,
        value=available_speakers[0],
        multiselect=False,
        visible = True,
    )
    ref_gr = gr.Audio(
        label="Reference Audio",
        # info="Click on the ✎ button to upload your own target speaker audio",
        type="filepath",
        # value="examples/female.wav",
        visible = False,
    )
    def visible_change(use_custom):
        return (
            gr.update(visible = not use_custom),
            gr.update(visible = use_custom)
        )
    use_custom_ref.change(visible_change, inputs=use_custom_ref, outputs=[ref_select_gr, ref_gr])
    clean_ref_gr = gr.Checkbox(
        label="Cleanup Reference Voice",
        value=True,
        info="This check can improve output if your microphone or reference voice is noisy",
    )
    auto_det_lang_gr = gr.Checkbox(
        label="Do not use language auto-detect",
        value=True,
        info="Check to disable language auto-detection",
    )
    tos_gr = gr.Checkbox(
        label="Agree",
        value=True,
        info="I have purchased a commercial license from Coqui: licensing@coqui.ai\nOtherwise, I agree to the terms of the non-commercial CPML: https://coqui.ai/cpml",
    )

    tts_button = gr.Button("Generate Speech from Text", elem_id="send-btn", visible=True)

    tts_button.click(
        tts_predict,
        [input_text_gr, language_gr, 
         use_custom_ref, ref_select_gr, ref_gr, 
         clean_ref_gr, auto_det_lang_gr, tos_gr],
        outputs=[output_audio_gr],
    )

    # with gr.Column():
    #     video_gr = gr.Video(label="Waveform Visual")
    #     audio_gr = gr.Audio(label="Synthesised Audio", autoplay=True)
    #     out_text_gr = gr.Text(label="Metrics")
    #     ref_audio_gr = gr.Audio(label="Reference Audio Used")

    # with gr.Row():
    #     gr.Examples(examples,
    #                 label="Examples",
    #                 inputs=[input_text_gr, language_gr, ref_gr, mic_gr, use_mic_gr, clean_ref_gr, auto_det_lang_gr, tos_gr],
    #                 outputs=[video_gr, audio_gr, out_text_gr, ref_audio_gr],
    #                 fn=predict,
    #                 cache_examples=False,)

    # tts_button.click(predict, [input_text_gr, language_gr, ref_gr, mic_gr, use_mic_gr, clean_ref_gr, auto_det_lang_gr, tos_gr], outputs=[video_gr, audio_gr, out_text_gr, ref_audio_gr])


if __name__ == "__main__":
    with gr.Blocks() as demo:
        audio_gr = gr.Audio(label="Synthesised Audio", autoplay=True)
        with gr.Column():
            make_tts_ui(audio_gr)
    demo.launch(server_port=10302)
