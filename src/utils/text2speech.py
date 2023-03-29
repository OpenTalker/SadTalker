import os

def text2speech(txt, audio_path):
    print(txt)
    cmd = f'tts --text "{txt}" --out_path {audio_path}'
    print(cmd)
    try:
        os.system(cmd)
        return audio_path
    except:
        print("Error: Failed convert txt to audio")
        return None