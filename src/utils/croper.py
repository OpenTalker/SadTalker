import os
import cv2
import time
import glob
import argparse
import scipy
import numpy as np
from PIL import Image
import torch
from tqdm import tqdm
from itertools import cycle

from src.face3d.extract_kp_videos_safe import KeypointExtractor
from facexlib.alignment import landmark_98_to_68

import numpy as np
from PIL import Image

class Preprocesser:
    def __init__(self, device='cuda'):
        self.predictor = KeypointExtractor(device)

    def get_landmark(self, img_np):
        """get landmark with dlib
        :return: np.array shape=(68, 2)
        """
        with torch.no_grad():
            dets = self.predictor.det_net.detect_faces(img_np, 0.97)

        if len(dets) == 0:
            return None
        det = dets[0]

        img = img_np[int(det[1]):int(det[3]), int(det[0]):int(det[2]), :]
        lm = landmark_98_to_68(self.predictor.detector.get_landmarks(img)) # [0]

        #### keypoints to the original location
        lm[:,0] += int(det[0])
        lm[:,1] += int(det[1])

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

        # Save aligned image.
        return rsize, crop, [lx, ly, rx, ry]
    
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
            if not still:
                _inp = _inp[ly:ry, lx:rx]
            img_np_list[_i] = _inp
        return img_np_list, crop, quad

