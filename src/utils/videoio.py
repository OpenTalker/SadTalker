import shutil
import uuid
import re
import os

import cv2

def sanitize_filename(filename, max_length=50):
    """
    Sanitize filename by removing/replacing special characters and limiting length.
    
    Args:
        filename (str): Original filename
        max_length (int): Maximum length for the filename
    
    Returns:
        str: Sanitized filename
    """
    # Remove file extension first
    name, ext = os.path.splitext(filename)
    
    # Replace special characters with underscores
    # Keep alphanumeric, spaces, hyphens, and underscores
    sanitized = re.sub(r'[^\w\s\-_.]', '_', name)
    
    # Replace multiple underscores with single underscore
    sanitized = re.sub(r'_+', '_', sanitized)
    
    # Remove leading/trailing underscores
    sanitized = sanitized.strip('_')
    
    # Limit length (accounting for extension)
    if len(sanitized) > max_length - len(ext):
        sanitized = sanitized[:max_length - len(ext)]
    
    # Ensure we have a valid filename
    if not sanitized:
        sanitized = "video"
    
    return sanitized + ext

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
    # Generate a unique temporary filename to avoid conflicts
    temp_file = str(uuid.uuid4())+'.mp4'
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # If the save_path already exists, remove it first to avoid conflicts
    if os.path.exists(save_path):
        try:
            os.remove(save_path)
        except OSError:
            # If we can't remove it, use a different temporary name
            temp_file = str(uuid.uuid4())+'_output.mp4'
    
    cmd = r'ffmpeg -y -hide_banner -loglevel error -i "%s" -i "%s" -vcodec copy "%s"' % (video, audio, temp_file)
    os.system(cmd)

    if watermark is False:
        # Move the temporary file to the final destination
        try:
            shutil.move(temp_file, save_path)
        except (OSError, shutil.Error) as e:
            # If move fails, try copy then remove
            try:
                shutil.copy2(temp_file, save_path)
                os.remove(temp_file)
            except Exception as copy_error:
                # If all else fails, just copy and leave temp file
                shutil.copy2(temp_file, save_path)
                print(f"Warning: Could not remove temporary file {temp_file}: {copy_error}")
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

        cmd = r'ffmpeg -y -hide_banner -loglevel error -i "%s" -i "%s" -filter_complex "[1]scale=100:-1[wm];[0][wm]overlay=(main_w-overlay_w)-10:10" "%s"' % (temp_file, watarmark_path, save_path)
        os.system(cmd)
        # Clean up temporary file
        try:
            os.remove(temp_file)
        except OSError:
            print(f"Warning: Could not remove temporary file {temp_file}")