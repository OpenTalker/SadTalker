"""run bash scripts/download_models.sh first to prepare the weights file"""
import os
import shutil
from argparse import Namespace
from src.utils.preprocess import CropAndExtract
from src.test_audio2coeff import Audio2Coeff
from src.facerender.animate import AnimateFromCoeff
from src.generate_batch import get_data
from src.generate_facerender_batch import get_facerender_data
from src.utils.init_path import init_path
from cog import BasePredictor, Input, Path

checkpoints = "checkpoints"


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        device = "cuda"

        
        sadtalker_paths = init_path(checkpoints,os.path.join("src","config"))

        # init model
        self.preprocess_model = CropAndExtract(sadtalker_paths, device
        )

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

    def predict(
        self,
        source_image: Path = Input(
            description="Upload the source image, it can be video.mp4 or picture.png",
        ),
        driven_audio: Path = Input(
            description="Upload the driven audio, accepts .wav and .mp4 file",
        ),
        enhancer: str = Input(
            description="Choose a face enhancer",
            choices=["gfpgan", "RestoreFormer"],
            default="gfpgan",
        ),
        preprocess: str = Input(
            description="how to preprocess the images",
            choices=["crop", "resize", "full"],
            default="full",
        ),
        ref_eyeblink: Path = Input(
            description="path to reference video providing eye blinking",
            default=None,
        ),
        ref_pose: Path = Input(
            description="path to reference video providing pose",
            default=None,
        ),
        still: bool = Input(
            description="can crop back to the original videos for the full body aniamtion when preprocess is full",
            default=True,
        ),
    ) -> Path:
        """Run a single prediction on the model"""

        animate_from_coeff = (
            self.animate_from_coeff["full"]
            if preprocess == "full"
            else self.animate_from_coeff["others"]
        )

        args = load_default()
        args.pic_path = str(source_image)
        args.audio_path = str(driven_audio)
        device = "cuda"
        args.still = still
        args.ref_eyeblink = None if ref_eyeblink is None else str(ref_eyeblink)
        args.ref_pose = None if ref_pose is None else str(ref_pose)

        # crop image and extract 3dmm from image
        results_dir = "results"
        if os.path.exists(results_dir):
            shutil.rmtree(results_dir)
        os.makedirs(results_dir)
        first_frame_dir = os.path.join(results_dir, "first_frame_dir")
        os.makedirs(first_frame_dir)

        print("3DMM Extraction for source image")
        first_coeff_path, crop_pic_path, crop_info = self.preprocess_model.generate(
            args.pic_path, first_frame_dir, preprocess, source_image_flag=True
        )
        if first_coeff_path is None:
            print("Can't get the coeffs of the input")
            return

        if ref_eyeblink is not None:
            ref_eyeblink_videoname = os.path.splitext(os.path.split(ref_eyeblink)[-1])[
                0
            ]
            ref_eyeblink_frame_dir = os.path.join(results_dir, ref_eyeblink_videoname)
            os.makedirs(ref_eyeblink_frame_dir, exist_ok=True)
            print("3DMM Extraction for the reference video providing eye blinking")
            ref_eyeblink_coeff_path, _, _ = self.preprocess_model.generate(
                ref_eyeblink, ref_eyeblink_frame_dir
            )
        else:
            ref_eyeblink_coeff_path = None

        if ref_pose is not None:
            if ref_pose == ref_eyeblink:
                ref_pose_coeff_path = ref_eyeblink_coeff_path
            else:
                ref_pose_videoname = os.path.splitext(os.path.split(ref_pose)[-1])[0]
                ref_pose_frame_dir = os.path.join(results_dir, ref_pose_videoname)
                os.makedirs(ref_pose_frame_dir, exist_ok=True)
                print("3DMM Extraction for the reference video providing pose")
                ref_pose_coeff_path, _, _ = self.preprocess_model.generate(
                    ref_pose, ref_pose_frame_dir
                )
        else:
            ref_pose_coeff_path = None

        # audio2ceoff
        batch = get_data(
            first_coeff_path,
            args.audio_path,
            device,
            ref_eyeblink_coeff_path,
            still=still,
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
            still_mode=still,
            preprocess=preprocess,
        )
        animate_from_coeff.generate(
            data, results_dir, args.pic_path, crop_info,
            enhancer=enhancer, background_enhancer=args.background_enhancer,
            preprocess=preprocess)

        output = "/tmp/out.mp4"
        mp4_path = os.path.join(results_dir, [f for f in os.listdir(results_dir) if "enhanced.mp4" in f][0])
        shutil.copy(mp4_path, output)

        return Path(output)


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
