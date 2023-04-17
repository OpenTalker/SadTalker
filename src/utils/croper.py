import os
import cv2
import time
import glob
import argparse
import scipy
import numpy as np
from PIL import Image
from tqdm import tqdm
from itertools import cycle

from torch.multiprocessing import Pool, Process, set_start_method


"""
brief: face alignment with FFHQ method (https://github.com/NVlabs/ffhq-dataset)
author: lzhbrian (https://lzhbrian.me)
date: 2020.1.5
note: code is heavily borrowed from 
    https://github.com/NVlabs/ffhq-dataset
    http://dlib.net/face_landmark_detection.py.html
requirements:
    apt install cmake
    conda install Pillow numpy scipy
    pip install dlib
    # download face landmark model from: 
    # http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
"""

import numpy as np
from PIL import Image
import dlib


class Croper:
    def __init__(self, path_of_lm):
        # download model from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
        self.predictor = dlib.shape_predictor(path_of_lm)

    def get_landmark(self, img_np):
        """get landmark with dlib
        :return: np.array shape=(68, 2)
        """
        detector = dlib.get_frontal_face_detector()
        dets = detector(img_np, 1)
        #     print("Number of faces detected: {}".format(len(dets)))
        #     for k, d in enumerate(dets):
        if len(dets) == 0:
            return None
        d = dets[0]
        # Get the landmarks/parts for the face in box d.
        shape = self.predictor(img_np, d)
        #         print("Part 0: {}, Part 1: {} ...".format(shape.part(0), shape.part(1)))
        t = list(shape.parts())
        a = []
        for tt in t:
            a.append([tt.x, tt.y])
        lm = np.array(a)
        # lm is a shape=(68,2) np.array
        return lm

    def align_face(self, img, lm, output_size=1024):
        """
        :param filepath: str
        :return: PIL Image
        """
        lm_chin = lm[0: 17]  # left-right
        lm_eyebrow_left = lm[17: 22]  # left-right
        lm_eyebrow_right = lm[22: 27]  # left-right
        lm_nose = lm[27: 31]  # top-down
        lm_nostrils = lm[31: 36]  # top-down
        lm_eye_left = lm[36: 42]  # left-clockwise
        lm_eye_right = lm[42: 48]  # left-clockwise
        lm_mouth_outer = lm[48: 60]  # left-clockwise
        lm_mouth_inner = lm[60: 68]  # left-clockwise

        # Calculate auxiliary vectors.
        eye_left = np.mean(lm_eye_left, axis=0)
        eye_right = np.mean(lm_eye_right, axis=0)
        eye_avg = (eye_left + eye_right) * 0.5
        eye_to_eye = eye_right - eye_left
        mouth_left = lm_mouth_outer[0]
        mouth_right = lm_mouth_outer[6]
        mouth_avg = (mouth_left + mouth_right) * 0.5
        eye_to_mouth = mouth_avg - eye_avg

        # Choose oriented crop rectangle.
        x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]  # Addition of binocular difference and double mouth difference
        x /= np.hypot(*x)   # hypot函数计算直角三角形的斜边长，用斜边长对三角形两条直边做归一化
        x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)    # 双眼差和眼嘴差，选较大的作为基准尺度
        y = np.flipud(x) * [-1, 1]
        c = eye_avg + eye_to_mouth * 0.1
        quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])   # 定义四边形，以面部基准位置为中心上下左右平移得到四个顶点
        qsize = np.hypot(*x) * 2    # 定义四边形的大小（边长），为基准尺度的2倍

        # Shrink.
        # 如果计算出的四边形太大了，就按比例缩小它
        shrink = int(np.floor(qsize / output_size * 0.5))
        if shrink > 1:
            rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
            img = img.resize(rsize, Image.ANTIALIAS)
            quad /= shrink
            qsize /= shrink
        else:
            rsize = (int(np.rint(float(img.size[0]))), int(np.rint(float(img.size[1]))))

        # Crop.
        border = max(int(np.rint(qsize * 0.1)), 3)
        crop = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
                int(np.ceil(max(quad[:, 1]))))
        crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]),
                min(crop[3] + border, img.size[1]))
        if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
            # img = img.crop(crop)
            quad -= crop[0:2]

        # Pad.
        pad = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
               int(np.ceil(max(quad[:, 1]))))
        pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - img.size[0] + border, 0),
               max(pad[3] - img.size[1] + border, 0))
        # if enable_padding and max(pad) > border - 4:
        #     pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
        #     img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
        #     h, w, _ = img.shape
        #     y, x, _ = np.ogrid[:h, :w, :1]
        #     mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w - 1 - x) / pad[2]),
        #                       1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h - 1 - y) / pad[3]))
        #     blur = qsize * 0.02
        #     img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
        #     img += (np.median(img, axis=(0, 1)) - img) * np.clip(mask, 0.0, 1.0)
        #     img = Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)), 'RGB')
        #     quad += pad[:2]

        # Transform.
        quad = (quad + 0.5).flatten()
        lx = max(min(quad[0], quad[2]), 0)
        ly = max(min(quad[1], quad[7]), 0)
        rx = min(max(quad[4], quad[6]), img.size[0])
        ry = min(max(quad[3], quad[5]), img.size[0])
        # img = img.transform((transform_size, transform_size), Image.QUAD, (quad + 0.5).flatten(),
        #                     Image.BILINEAR)
        # if output_size < transform_size:
        #     img = img.resize((output_size, output_size), Image.ANTIALIAS)

        # Save aligned image.
        return rsize, crop, [lx, ly, rx, ry]

    # def crop(self, img_np_list):
    #     for _i in range(len(img_np_list)):
    #         img_np = img_np_list[_i]
    #         lm = self.get_landmark(img_np)
    #         if lm is None:
    #             return None
    #         crop, quad = self.align_face(img=Image.fromarray(img_np), lm=lm, output_size=512)
    #         clx, cly, crx, cry = crop
    #         lx, ly, rx, ry = quad
    #         lx, ly, rx, ry = int(lx), int(ly), int(rx), int(ry)
        
    #         _inp = img_np_list[_i]
    #         _inp = _inp[cly:cry, clx:crx]
    #         _inp = _inp[ly:ry, lx:rx]
    #         img_np_list[_i] = _inp
    #     return img_np_list
    
    def crop(self, img_np_list, still=False, xsize=512):    # first frame for all video
        img_np = img_np_list[0]
        lm = self.get_landmark(img_np)
        if lm is None:
            raise 'can not detect the landmark from source image'
        rsize, crop, quad = self.align_face(img=Image.fromarray(img_np), lm=lm, output_size=xsize)
        clx, cly, crx, cry = crop
        lx, ly, rx, ry = quad
        lx, ly, rx, ry = int(lx), int(ly), int(rx), int(ry)
        for _i in range(len(img_np_list)):
            _inp = img_np_list[_i]
            _inp = cv2.resize(_inp, (rsize[0], rsize[1]))
            _inp = _inp[cly:cry, clx:crx]
            # cv2.imwrite('test1.jpg', _inp)
            if not still:
                _inp = _inp[ly:ry, lx:rx]
            # cv2.imwrite('test2.jpg', _inp)
            img_np_list[_i] = _inp
        return img_np_list, crop, quad


def read_video(filename, uplimit=100):
    frames = []
    cap = cv2.VideoCapture(filename)
    cnt = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (512, 512))
            frames.append(frame)
        else:
            break
        cnt += 1
        if cnt >= uplimit:
            break
    cap.release()
    assert len(frames) > 0, f'{filename}: video with no frames!'
    return frames


def create_video(video_name, frames, fps=25, video_format='.mp4', resize_ratio=1):
    # video_name = os.path.dirname(image_folder) + video_format
    # img_list = glob.glob1(image_folder, 'frame*')
    # img_list.sort()
    # frame = cv2.imread(os.path.join(image_folder, img_list[0]))
    # frame = cv2.resize(frame, (0, 0), fx=resize_ratio, fy=resize_ratio)
    # height, width, layers = frames[0].shape
    height, width, layers = 512, 512, 3
    if video_format == '.mp4':
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    elif video_format == '.avi':
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video = cv2.VideoWriter(video_name, fourcc, fps, (width, height))
    for _frame in frames:
        _frame = cv2.resize(_frame, (height, width), interpolation=cv2.INTER_LINEAR)
        video.write(_frame)

def create_images(video_name, frames):
    height, width, layers = 512, 512, 3
    images_dir = video_name.split('.')[0]
    os.makedirs(images_dir, exist_ok=True)
    for i, _frame in enumerate(frames):
        _frame = cv2.resize(_frame, (height, width), interpolation=cv2.INTER_LINEAR)
        _frame_path = os.path.join(images_dir, str(i)+'.jpg')
        cv2.imwrite(_frame_path, _frame)

def run(data):
    filename, opt, device = data
    os.environ['CUDA_VISIBLE_DEVICES'] = device
    croper = Croper()

    frames = read_video(filename, uplimit=opt.uplimit)
    name = filename.split('/')[-1]  # .split('.')[0]
    name = os.path.join(opt.output_dir, name)

    frames = croper.crop(frames)
    if frames is None:
        print(f'{name}: detect no face. should removed')
        return
    # create_video(name, frames)
    create_images(name, frames)


def get_data_path(video_dir):
    eg_video_files = ['/apdcephfs/share_1290939/quincheng/datasets/HDTF/backup_fps25/WDA_KatieHill_000.mp4']
    # filenames = list()
    # VIDEO_EXTENSIONS_LOWERCASE = {'mp4'}
    # VIDEO_EXTENSIONS = VIDEO_EXTENSIONS_LOWERCASE.union({f.upper() for f in VIDEO_EXTENSIONS_LOWERCASE})
    # extensions = VIDEO_EXTENSIONS
    # for ext in extensions:
    #     filenames = sorted(glob.glob(f'{opt.input_dir}/**/*.{ext}'))
    # print('Total number of videos:', len(filenames))
    return eg_video_files


def get_wra_data_path(video_dir):
    if opt.option == 'video':
        videos_path = sorted(glob.glob(f'{video_dir}/*.mp4'))
    elif opt.option == 'image':
        videos_path = sorted(glob.glob(f'{video_dir}/*/'))
    else:
        raise NotImplementedError
    print('Example videos: ', videos_path[:2])
    return videos_path


if __name__ == '__main__':
    set_start_method('spawn')
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input_dir', type=str, help='the folder of the input files')
    parser.add_argument('--output_dir', type=str, help='the folder of the output files')
    parser.add_argument('--device_ids', type=str, default='0,1')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--uplimit', type=int, default=500)
    parser.add_argument('--option', type=str, default='video')

    root = '/apdcephfs/share_1290939/quincheng/datasets/HDTF'
    cmd = f'--input_dir {root}/backup_fps25_first20s_sync/ ' \
          f'--output_dir {root}/crop512_stylegan_firstframe_sync/ ' \
          '--device_ids 0 ' \
          '--workers 8 ' \
          '--option video ' \
          '--uplimit 500 '
    opt = parser.parse_args(cmd.split())
    # filenames = get_data_path(opt.input_dir)
    filenames = get_wra_data_path(opt.input_dir)
    os.makedirs(opt.output_dir, exist_ok=True)
    print(f'Video numbers: {len(filenames)}')
    pool = Pool(opt.workers)
    args_list = cycle([opt])
    device_ids = opt.device_ids.split(",")
    device_ids = cycle(device_ids)
    for data in tqdm(pool.imap_unordered(run, zip(filenames, args_list, device_ids))):
        None
