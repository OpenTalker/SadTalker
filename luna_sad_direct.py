# 这里面的import都是外部库，和sad、fay、xuniren无关
# 使用stable_nlp虚拟环境运行
import time
import os
import numpy as np
import threading
from pydub import AudioSegment
import cv2
import pygame
import hashlib
from fastapi import FastAPI
import uvicorn
from pathlib import Path



face_path = r"E:/AsciiStandardPath/InnovativeExploration/CFAR/girl_face.jpeg"
待机动画 = r'E:\AsciiStandardPath\InnovativeExploration\CFAR\SadTalker\data\girl_face待机动画1.mp4'
music_directory = Path(r'C:\Users\YeCanming\Music\ ').resolve()
直播室背景音乐 = [x.as_posix() for x in music_directory.glob('*.mp3')]
# 直播室背景音乐 = [f"{music_directory.as_posix()}/【P1】AuntTagonist(即通辽进行曲)_20240122_01022800.mp3", 
#            ]
import random
random.shuffle(直播室背景音乐)

crop_size = 256 # 不能是str
import tempfile
import shutil

temp_dir = tempfile.TemporaryDirectory()


from src.gradio_demo import SadTalker
sad_talker = SadTalker(checkpoint_path="checkpoints", config_path="src/config", lazy_load=True)
# from gradio_client import Client
# host = "http://127.0.0.1"
# # port = 12746
# port = 13462
# client = Client(f'{host}:{port}', verbose=True, 
#                 serialize=True)

args = dict(
        source_image=None, 
        driven_audio=None, preprocess='crop', 
        still_mode=True,  use_enhancer=False, 
        # batch_size=2, 
        batch_size=8, 
        size=crop_size, 
        pose_style = 0, 
        # exp_scale=1.0, 
        # use_ref_video = False,
        # ref_video = None,
        # ref_info = None,
        # use_idle_mode = False,
        # length_of_audio = 0, use_blink=True,
        # result_dir='./results/'
)
def make_copy(path):
    try:
        shutil.copy(path, temp_dir.name)
    except shutil.SameFileError:
        pass
    path = os.path.join(temp_dir.name, os.path.basename(path))
    return path
# async def generate_video(audio_path, output_path=None):
def generate_video(audio_path, output_path=None):
    global face_path
    local_face_path = make_copy(face_path)
    audio_path = make_copy(audio_path)
    
    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    args.update(
        dict(
            source_image=local_face_path,
            driven_audio=audio_path,
            result_dir=temp_dir.name
        )
    )
    result = sad_talker.test(**args)
    # result = await sad_talker.test(**args)
    result_path = result
    # result = client.predict(
    #         **args, driven_audio=audio_path, 
    #         api_name="/test"
    #     )
    # result_path = result['video']
    
    if output_path is not None:
        shutil.move(result_path, output_path)
        return output_path
    return result_path

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

# https://www.pygame.org/docs/ref/mixer.html
pygame.mixer.init()
pygame.mixer.set_num_channels(8)
def play_audio(audio_file, volume=1.0, loops=0):
    # The loops argument controls how many times the sample will be repeated after being played the first time.
    # 注意loops是额外的次数，而不是总共的次数。所以0表示播放1次。-1是特殊情况，无限循环。
    sound = pygame.mixer.Sound(audio_file)
    sound.set_volume(volume)
    sound.play(loops=loops)
    while pygame.mixer.get_busy():
        pygame.time.Clock().tick(10)
    
    
def play_video():
    """不断读取video_list, 播放最新视频。如果没有视频，就播放train.mp4的第一帧。
    """
    global video_list
    video_path = None
    audio_path = None
    ret = None
    frame = None
    try:
        while True:
            # 找到要播放什么
            is_not_idle = (len(video_list) > 0)
            if is_not_idle:
                video_path = video_list[0].get("video")
                audio_path = video_list[0].get("audio")
                cap = cv2.VideoCapture(video_path)  # 打开视频文件
                video_list.pop(0)
            else:
                audio_path = None
                # cap = None
                # _, frame = cv2.VideoCapture(r"E:/AsciiStandardPath/InnovativeExploration/CFAR/xuniren/data/pretrained/train.mp4").read()
                # _, frame = cv2.VideoCapture("results/ff384bb2-a890-4002-91dc-1fb581893b83/girl_face##sustcutech_10.mp4").read()
                # frame = cv2.imread(face_path)
                # frame = cv2.resize(frame, (256, 256))

                cap = cv2.VideoCapture(待机动画)  

            # 后台播放音频
            if audio_path:
                threading.Thread(target=play_audio, args=[audio_path]).start()  # play audio
            # 循环播放视频帧
            while True:
                if not is_not_idle and (len(video_list) > 0):
                    break # 跳出待机动画的循环
                if cap:
                    ret, frame = cap.read()
                if frame is not None:  # 没有传音频过来时显示train.mp4的第一帧，建议替换成大约1秒左右的视频
                    cv2.imshow("SadTalker", frame)
                    # 等待 38 毫秒
                    cv2.waitKey(38) # 帧率控制
                if not ret:
                    break
    except InterruptedError as e:
        print(f"play_video interrupted: {e}")
    finally:
        cv2.destroyAllWindows()



from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Setup code
    threading.Thread(target=play_video, 
                          daemon=True).start()
    
    def play_background():
        while True:
            for bgm in 直播室背景音乐:
                print("播放背景音乐：", bgm)
                play_audio(bgm, 
                        #    volume=0.1
                           volume=0.05
                           )
                print("播放完毕")
                time.sleep(1)
    threading.Thread(target=play_background,  
                     daemon=True).start()
    
    print("FastAPI application is starting. ")
    yield
    print("FastAPI application is shutting down. ")

    # Teardown code
    # t1.join() # daemon=True 那就不用join了
    
# 2. FastAPI 有关功能
# https://fastapi.tiangolo.com/advanced/events/#alternative-events-deprecated
app = FastAPI(lifespan=lifespan)
def start_fastapi_server():
    uvicorn.run(app, host="127.0.0.1", port=18800, 
                )

import asyncio
from concurrent.futures import ThreadPoolExecutor
executor = ThreadPoolExecutor(max_workers=2)
# asyncio.get_event_loop().set_default_executor(executor)
# app.state.executor = executor
    
# 3. 全局状态
video_list: list[
    dict[str, str]
] = []  # 是 {"video": str, "audio": str} 的列表, 表示已经生成过的视频在哪里

# fay_ws = None
video_cache: dict[str, str] = {}  # 是 {"audio_hash": "video_path"} 的字典，根据md5推算出来的视频在哪里

@app.get("/audio_to_video/")
async def audio_to_video_async(file_path: str):
    return await asyncio.get_event_loop().run_in_executor(executor, audio_to_video, (file_path, ))

def audio_to_video(file_path: str):
    
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
        audio_path = new_path
        # audio_process(audio_path)
        # audio_path_eo = "data/audio/aud_%d_eo.npy" % num
        # video_path = "data/video/results/ngp_%d.mp4" % num
        output_path = "data/video/results/output_%d.mp4" % num
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # generate_video(audio_path, audio_path_eo, video_path, output_path)
        # output_path = await generate_video(audio_path, output_path)
        output_path = generate_video(audio_path, output_path)
    
        video_list.append({"video": output_path, "audio": new_path})
        video_cache[audio_hash] = output_path

    return {"code": 200}


if __name__ == "__main__":
    start_fastapi_server()
    
    


# import threading, time, signal, sys

# class StoppableThread(threading.Thread):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.stop_flag = False
#         self.daemon = True
#     def stop(self):
#         self.stop_flag = True
# StoppableThread = threading.Thread
# StoppableThread.stop = threading.Thread._stop
# StoppableThread.stop_flag = threading.Thread._is_stopped

# @app.on_event("shutdown")
# def shutdown_event():
#     # 在这里添加你的清理逻辑
#     print("FastAPI application is shutting down。 ")
# def signal_handler(signal, frame):
#     print(f"Caught signal {signal} in frame {frame}.")
#     for thread in threading.enumerate():
#         # if isinstance(thread, StoppableThread):
#             thread._stop()
#     print("All threads are stopped. ")
#     sys.exit(0)
#     print("sys.exit(0) is called. ")
    
# if __name__ == "__main__":
#     # 创建并启动FastAPI服务的线程
#     signal.signal(signal.SIGINT, signal_handler)
#     signal.signal(signal.SIGTERM, signal_handler)
#     threads:list[StoppableThread] = [
#         StoppableThread(target = start_fastapi_server),
#         StoppableThread(target = play_video)
#     ]
#     for thread in threads:
#         thread.start()
#     try:
#         for thread in threads:
#             thread.join()
#     except KeyboardInterrupt as e:
#         # 这个优先级没有signal_handler高，所以应该不会被调用
#         print(f"Caught KeyboardInterrupt {e}. ")
#         for thread in threads:
#             thread.stop()
            
    # daemon的优先级也不高，整个退出的时候才会被调用
    
# 多线程有关的知识
# 1. quit用于解释器、exit用于脚本。sys.exit()和quit()函数只能结束调用它们的线程，不能结束其他线程。
# 2. daemon线程是一种支持线程，当主线程退出时，daemon线程也会退出，不管是否执行完任务。
# 3. 信号只能被主线程接收和处理，子线程无法接收信号。
# 4. signal.SIGTERM是kill命令发送给进程的默认信号，signal.SIGINT是Ctrl+C发送给进程的默认信号。
# 5. Windows支持SIGINT和SIGTERM。
# 6. python的 thread 不能 stop
# 7. is_alive 一般在外面判断，里面判断基本都是alive的，因为判断的时候线程还没结束。