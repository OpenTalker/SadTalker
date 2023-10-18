"""
run following commands first to download weights:
    1. bash scripts/download_models.sh
    2. wget --content-disposition https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth --directory gfpgan/weights
    3. wget --content-disposition https://github.com/xinntao/facexlib/releases/download/v0.1.0/alignment_WFLW_4HG.pth --directory gfpgan/weights
"""
import os
import shutil
from argparse import Namespace
from typing import List, Optional

from tungstenkit import Audio, BaseIO, Field, Image, Option, Video, define_model

from src.facerender.animate import AnimateFromCoeff
from src.generate_batch import get_data
from src.generate_facerender_batch import get_facerender_data
from src.test_audio2coeff import Audio2Coeff
from src.utils.init_path import init_path
from src.utils.preprocess import CropAndExtract


class Input(BaseIO):
    source_image: Image = Field(
        description="Upload the source image.",
    )
    driven_audio: Audio = Field(
        description="Upload the driven audio, accepts .wav and .mp4 file",
    )
    ref_eyeblink: Optional[Video] = Option(
        description="path to reference video providing eye blinking",
        default=None,
    )
    ref_pose: Optional[Video] = Option(
        description="path to reference video providing pose",
        default=None,
    )


class Output(BaseIO):
    output_video: Video


checkpoints = "checkpoints"


@define_model(
    input=Input,
    output=Output,
    gpu=True,
    gpu_mem_gb=15,
    system_packages=[
        "ffmpeg",
        "libgl1-mesa-glx",
        "libglib2.0-0",
        "wget",
    ],
    python_packages=[
        "torch==1.12.1",
        "torchvision==0.13.1",
        "torchaudio==0.12.1",
        "joblib==1.1.0",
        "scikit-image==0.19.3",
        "basicsr==1.4.2",
        "facexlib==0.3.0",
        "resampy==0.3.1",
        "pydub==0.25.1",
        "scipy==1.10.1",
        "kornia==0.6.8",
        "face_alignment==1.3.5",
        "imageio==2.19.3",
        "imageio-ffmpeg==0.4.7",
        "librosa==0.9.2",
        "tqdm==4.65.0",
        "yacs==0.1.8",
        "gfpgan==1.3.8",
        "dlib-bin==19.24.1",
        "av==10.0.0",
        "trimesh==3.9.20",
        "numpy==1.23.4",
        "safetensors",
    ],
    dockerfile_commands=[
        'RUN mkdir -p /root/.cache/torch/hub/checkpoints/ && wget --output-document "/root/.cache/torch/hub/checkpoints/s3fd-619a316812.pth" "https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth"',
        'RUN mkdir -p /root/.cache/torch/hub/checkpoints/ && wget --output-document "/root/.cache/torch/hub/checkpoints/2DFAN4-cd938726ad.zip" "https://www.adrianbulat.com/downloads/python-fan/2DFAN4-cd938726ad.zip"',
    ],
    python_version="3.8",
    cuda_version="11.3",
)
class SadTalkerModel:
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        device = "cuda"

        sadtalker_paths = init_path(
            checkpoints, os.path.join("src", "config"), 256, False, "full"
        )

        # init model
        self.preprocess_model = CropAndExtract(sadtalker_paths, device)

        self.audio_to_coeff = Audio2Coeff(
            sadtalker_paths,
            device,
        )

        self.animate_from_coeff = {
            "full": AnimateFromCoeff(
                sadtalker_paths,
                device,
            ),
            "others": AnimateFromCoeff(
                sadtalker_paths,
                device,
            ),
        }

    def predict(self, inputs: List[Input]) -> List[Output]:
        """Run a single prediction on the model"""

        input = inputs[0]

        animate_from_coeff = self.animate_from_coeff["full"]

        args = load_default()
        args.pic_path = str(input.source_image.path)
        args.audio_path = str(input.driven_audio.path)
        device = "cuda"
        args.still = True
        args.ref_eyeblink = (
            None if input.ref_eyeblink is None else str(input.ref_eyeblink.path)
        )
        args.ref_pose = None if input.ref_pose is None else str(input.ref_pose.path)

        # crop image and extract 3dmm from image
        results_dir = "results"
        if os.path.exists(results_dir):
            shutil.rmtree(results_dir)
        os.makedirs(results_dir)
        first_frame_dir = os.path.join(results_dir, "first_frame_dir")
        os.makedirs(first_frame_dir)

        print("3DMM Extraction for source image")
        first_coeff_path, crop_pic_path, crop_info = self.preprocess_model.generate(
            args.pic_path, first_frame_dir, "full", source_image_flag=True
        )
        if first_coeff_path is None:
            print("Can't get the coeffs of the input")
            return

        if input.ref_eyeblink is not None:
            ref_eyeblink_videoname = os.path.splitext(
                os.path.split(input.ref_eyeblink.path)[-1]
            )[0]
            ref_eyeblink_frame_dir = os.path.join(results_dir, ref_eyeblink_videoname)
            os.makedirs(ref_eyeblink_frame_dir, exist_ok=True)
            print("3DMM Extraction for the reference video providing eye blinking")
            ref_eyeblink_coeff_path, _, _ = self.preprocess_model.generate(
                input.ref_eyeblink.path, ref_eyeblink_frame_dir
            )
        else:
            ref_eyeblink_coeff_path = None

        if input.ref_pose is not None:
            if input.ref_pose == input.ref_eyeblink:
                ref_pose_coeff_path = ref_eyeblink_coeff_path
            else:
                ref_pose_videoname = os.path.splitext(
                    os.path.split(input.ref_pose.path)[-1]
                )[0]
                ref_pose_frame_dir = os.path.join(results_dir, ref_pose_videoname)
                os.makedirs(ref_pose_frame_dir, exist_ok=True)
                print("3DMM Extraction for the reference video providing pose")
                ref_pose_coeff_path, _, _ = self.preprocess_model.generate(
                    input.ref_pose.path, ref_pose_frame_dir
                )
        else:
            ref_pose_coeff_path = None

        # audio2ceoff
        batch = get_data(
            first_coeff_path,
            args.audio_path,
            device,
            ref_eyeblink_coeff_path,
            still=True,
        )
        coeff_path = self.audio_to_coeff.generate(
            batch, results_dir, args.pose_style, ref_pose_coeff_path
        )
        # coeff2video
        print("coeff2video")
        data = get_facerender_data(
            coeff_path,
            crop_pic_path,
            first_coeff_path,
            args.audio_path,
            args.batch_size,
            args.input_yaw,
            args.input_pitch,
            args.input_roll,
            expression_scale=args.expression_scale,
            still_mode=True,
            preprocess="full",
        )
        animate_from_coeff.generate(
            data,
            results_dir,
            args.pic_path,
            crop_info,
            enhancer="gfpgan",
            background_enhancer=args.background_enhancer,
            preprocess="full",
        )

        output = "output.mp4"
        mp4_path = os.path.join(
            results_dir, [f for f in os.listdir(results_dir) if "enhanced.mp4" in f][0]
        )
        shutil.copy(mp4_path, output)

        return [Output(output_video=Video.from_path(output))]


def load_default():
    return Namespace(
        pose_style=0,
        batch_size=2,
        expression_scale=1.0,
        input_yaw=None,
        input_pitch=None,
        input_roll=None,
        background_enhancer=None,
        face3dvis=False,
        net_recon="resnet50",
        init_path=None,
        use_last_fc=False,
        bfm_folder="./src/config/",
        bfm_model="BFM_model_front.mat",
        focal=1015.0,
        center=112.0,
        camera_d=10.0,
        z_near=5.0,
        z_far=15.0,
    )
