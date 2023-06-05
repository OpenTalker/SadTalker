import os
import tempfile
from TTS.api import TTS


class TTSTalker():
    def __init__(self) -> None:
        model_name = TTS.list_models()[0]
        self.tts = TTS(model_name)

    def test(self, text, language='en'):

        tempf  = tempfile.NamedTemporaryFile(
                delete = False,
                suffix = ('.'+'wav'),
            )

        self.tts.tts_to_file(text, speaker=self.tts.speakers[0], language=language, file_path=tempf.name)

        return tempf.name