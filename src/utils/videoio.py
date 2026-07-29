import shutil
import subprocess
import uuid

import os

import cv2

def load_video_to_cv2(input_path):
    video_stream = cv2.VideoCapture(input_path)
    fps = video_stream.get(cv2.CAP_PROP_FPS)
    full_frames = [] 
    while 1:
        still_reading, frame = video_stream.read()
        if not still_reading:
            video_stream.release()
            break 
        full_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return full_frames

def save_video_with_watermark(video, audio, save_path, watermark=False):
    temp_file = str(uuid.uuid4())+'.mp4'
    # Use argument list (no shell) to avoid OS command injection via file paths
    # that may contain shell metacharacters (CWE-78).
    subprocess.run(
        ['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
         '-i', video, '-i', audio, '-vcodec', 'copy', temp_file],
        check=True,
    )

    if watermark is False:
        shutil.move(temp_file, save_path)
    else:
        # watermark
        try:
            ##### check if stable-diffusion-webui
            import webui
            from modules import paths
            watarmark_path = paths.script_path+"/extensions/SadTalker/docs/sadtalker_logo.png"
        except:
            # get the root path of sadtalker.
            dir_path = os.path.dirname(os.path.realpath(__file__))
            watarmark_path = dir_path+"/../../docs/sadtalker_logo.png"

        subprocess.run(
            ['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
             '-i', temp_file, '-i', watarmark_path,
             '-filter_complex',
             '[1]scale=100:-1[wm];[0][wm]overlay=(main_w-overlay_w)-10:10',
             save_path],
            check=True,
        )
        os.remove(temp_file)
