# 这里面的import都是外部库，和sad、fay、xuniren无关
# 使用gpt虚拟环境运行
import base64
import time
import json
# import gevent
# from gevent import pywsgi
# from geventwebsocket.handler import WebSocketHandler
import os
import re
import numpy as np
import threading
import websocket
from pydub import AudioSegment
# from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_videoclips
import cv2
import pygame
import hashlib
from fastapi import FastAPI
import uvicorn

# from tools import audio_pre_process, video_pre_process, generate_video, audio_process

from gradio_client import Client
host = "http://127.0.0.1"
# port = 12746
port = 13462
client = Client(f'{host}:{port}', verbose=True, 
                serialize=True)
        
face_path = r"E:/AsciiStandardPath/InnovativeExploration/CFAR/girl_face.jpeg"
crop_size = 256 # 不能是str
# 1. 音频基础处理函数
# 增加MD5音频标记，避免重复生成视频
def hash_file_md5(filepath):
    md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        while True:
            data = f.read(65536)  # Read in 64kb chunks
            if not data:
                break
            md5.update(data)
    return md5.hexdigest()


def convert_mp3_to_wav(input_file, output_file):
    print(f"Converting {input_file} to {output_file}")
    try:
        # 检查文件是否存在
        if not os.path.isfile(input_file):
            raise FileNotFoundError("Input file does not exist.")

        # 检查文件格式
        audio_extension = os.path.splitext(input_file)[1].lower()
        if audio_extension not in [".mp3", ".wav"]:
            raise ValueError("Unsupported file format")

        # 读取音频文件
        # 不是Python的bug，是缺了目录, 需要创建一个audio文件夹
        from pathlib import Path

        Path(input_file).parent.mkdir(parents=True, exist_ok=True)
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)

        audio = AudioSegment.from_file(input_file)

        # 导出为 WAV 格式
        audio.export(output_file, format="wav")
        print(f"File converted and saved as {output_file}")
    except FileNotFoundError as e:
        raise e
        print(f"FileNotFoundError at convert_mp3_to_wav: {e}")
    except ValueError as e:
        raise e
        print(f"Error: {e}")

def play_audio(audio_file):
    pygame.mixer.init()
    pygame.mixer.music.load(audio_file)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)
    pygame.mixer.music.stop()
    
# 2. FastAPI 有关功能
app = FastAPI()
def start_fastapi_server():
    uvicorn.run(app, host="0.0.0.0", port=18800)
    
# 3. 全局状态
video_list: list[
    dict[str, str]
] = []  # 是 {"video": str, "audio": str} 的列表, 表示已经生成过的视频在哪里

# fay_ws = None
video_cache: dict[str, str] = {}  # 是 {"audio_hash": "video_path"} 的字典，根据md5推算出来的视频在哪里

@app.get("/audio_to_video/")
async def audio_to_video(file_path: str):
    print("file_path=", file_path)
    aud_dir = file_path
    aud_dir = aud_dir.replace("\\", "/")
    print("message:", aud_dir, type(aud_dir))
    basedir = ""
    for i in aud_dir.split("/"):
        basedir = os.path.join(basedir, i)
    basedir = basedir.replace(":", ":\\")
    num = time.time()
    new_path = r"./data/audio/aud_%d.wav" % num  # 新路径
    old_path = basedir
    # old_path = file_path

    convert_mp3_to_wav(old_path, new_path)
    audio_hash = hash_file_md5(new_path)
    if audio_hash in video_cache:
        video_list.append({"video": video_cache[audio_hash], "audio": new_path})
        print("视频已存在，直接播放。")
    else:
        audio_path = "data/audio/aud_%d.wav" % num
        # audio_process(audio_path)
        # audio_path_eo = "data/audio/aud_%d_eo.npy" % num
        # video_path = "data/video/results/ngp_%d.mp4" % num
        output_path = "data/video/results/output_%d.mp4" % num

        
        # generate_video(audio_path, audio_path_eo, video_path, output_path)
        result = client.predict(
            # "https://raw.githubusercontent.com/gradio-app/gradio/main/test/test_files/bus.png",	# filepath  in 'Source image' Image component
            # "https://github.com/gradio-app/gradio/raw/main/test/test_files/audio_sample.wav",	# filepath  in 'Input audio' Audio component
            face_path, 
            # r"E:/AsciiStandardPath/InnovativeExploration/CFAR/Let's Go.mp3", 
            audio_path, 
            "crop",	# Literal['crop', 'resize', 'full', 'extcrop', 'extfull']  in 'preprocess' Radio component
            True,	# bool  in 'Still Mode (fewer head motion, works with preprocess `full`)' Checkbox component
            False,	# bool  in 'GFPGAN as Face enhancer' Checkbox component
            10,	# float (numeric value between 0 and 10) in 'batch size in generation' Slider component
            crop_size,	# Literal['256', '512']  in 'face model resolution' Radio component
            0,	# float (numeric value between 0 and 46) in 'Pose style' Slider component
            api_name="/test"
        )
        # print(result)
        

        # video_list.append({"video": output_path, "audio": new_path})
        video_list.append({"video": result['video'], "audio": new_path})
        video_cache[audio_hash] = output_path

    return {"code": 200}





def play_video():
    """不断读取video_list, 播放最新视频。如果没有视频，就播放train.mp4的第一帧。
    """
    global video_list
    video_path = None
    audio_path = None
    ret = None
    frame = None
    while threading.current_thread().is_alive():
        # 找到要播放什么
        if len(video_list) > 0:
            video_path = video_list[0].get("video")
            audio_path = video_list[0].get("audio")
            cap = cv2.VideoCapture(video_path)  # 打开视频文件
            video_list.pop(0)
        else:
            audio_path = None
            cap = None
            # _, frame = cv2.VideoCapture(r"E:/AsciiStandardPath/InnovativeExploration/CFAR/xuniren/data/pretrained/train.mp4").read()
            frame = cv2.imread(face_path)
            frame = cv2.resize(frame, (256, 256))

        # 后台播放音频
        if audio_path:
            threading.Thread(target=play_audio, args=[audio_path]).start()  # play audio
        # 循环播放视频帧
        while threading.current_thread().is_alive():
            if cap:
                ret, frame = cap.read()
            if frame is not None:  # 没有传音频过来时显示train.mp4的第一帧，建议替换成大约1秒左右的视频
                cv2.imshow("Fay-2d", frame)
                # 等待 38 毫秒
                cv2.waitKey(38) # 帧率控制
            if not ret:
                break
    cv2.destroyAllWindows()



import threading, time, signal

if __name__ == "__main__":
    # audio_pre_process()
    # video_pre_process()

    # 创建并启动FastAPI服务的线程
    try:
        signal.signal(signal.SIGINT, quit)
        signal.signal(signal.SIGTERM, quit)

        a = threading.Thread(target = start_fastapi_server)
        b = threading.Thread(target = play_video)
        a.daemon = True
        b.daemon = True
        a.start()
        b.start()

        a.join()
        b.join()
    except InterruptedError as exc:
        print(exc)
