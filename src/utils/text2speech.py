import os
import tempfile
from TTS.api import TTS


class TTSTalker():
    def __init__(self) -> None:
        model_name = TTS().list_models()[0]
        self.tts = TTS(model_name)

    def test(self, text, audio, language='en'):

        tempf  = tempfile.NamedTemporaryFile(
                delete = False,
                suffix = ('.'+'wav'),
            )

        self.tts.tts_to_file(text, speaker_wav=audio, language=language, file_path=tempf.name)

        return tempf.name
