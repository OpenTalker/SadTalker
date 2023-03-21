<div align="center">

<h2> 😭 SadTalker： <span style="font-size:12px">Learning Realistic 3D Motion Coefficients for  Stylized Audio-Driven Single Image Talking Face Animation </span> </h2> 

  <a href='https://arxiv.org/abs/2211.12194'><img src='https://img.shields.io/badge/ArXiv-2211.14758-red'></a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href='https://sadtalker.github.io'><img src='https://img.shields.io/badge/Project-Page-Green'></a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Winfredy/SadTalker/blob/main/quick_demo.ipynb) 

<div>
    <a target='_blank'>Wenxuan Zhang <sup>*,1,2</sup> </a>&emsp;
    <a href='https://vinthony.github.io/' target='_blank'>Xiaodong Cun <sup>*,2</a>&emsp;
    <a href='https://xuanwangvc.github.io/' target='_blank'>Xuan Wang <sup>3</sup></a>&emsp;
    <a href='https://yzhang2016.github.io/' target='_blank'>Yong Zhang <sup>2</sup></a>&emsp;
    <a href='https://xishen0220.github.io/' target='_blank'>Xi Shen <sup>2</sup></a>&emsp; </br>
    <a href='https://yuguo-xjtu.github.io/' target='_blank'>Yu Guo<sup>1</sup> </a>&emsp;
    <a href='https://scholar.google.com/citations?hl=zh-CN&user=4oXBp9UAAAAJ' target='_blank'>Ying Shan <sup>2</sup> </a>&emsp;
    <a target='_blank'>Fei Wang <sup>1</sup> </a>&emsp;
</div>
<br>
<div>
    <sup>1</sup> Xi'an Jiaotong University &emsp; <sup>2</sup> Tencent AI Lab &emsp; <sup>3</sup> Ant Group &emsp; 
</div>
<br>
<i><strong><a href='https://arxiv.org/abs/2211.12194' target='_blank'>CVPR 2023</a></strong></i>
<br>
<br>

![sadtalker](https://user-images.githubusercontent.com/4397546/222490039-b1f6156b-bf00-405b-9fda-0c9a9156f991.gif)

<b>TL;DR: A realistic and stylized talking head video generation method from a single image and audio.</b>

<br>

</div>


## 📋 Changelog

- 2023.03.18 Support `expression intensity`, now you can change the intensity of the generated motion: `python inference.py --expression_scale 2(some value > 1)`.

- 2023.03.18 Reconfig the data folders, now you can download the checkpoint automatically using `bash utils/download_models.sh`.
- 2023.03.18 We have offically integrate the [GFPGAN](https://github.com/TencentARC/GFPGAN) for face enhancement, using `python inference.py --enhancer gfpgan` for  better visualization performance.
- 2023.03.14 Specify the version of package `joblib` to remove the errors in using `librosa`, [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Winfredy/SadTalker/blob/main/quick_demo.ipynb) is online!
&nbsp;&nbsp;&nbsp;&nbsp; <details><summary> Previous Changelogs</summary>
  - 2023.03.06 Solve some bugs in code and errors in installation 
  - 2023.03.03 Release the test code for audio-driven single image animation!
  - 2023.02.28 SadTalker has been accepted by CVPR 2023!

</details>

## 🎼 Pipeline
![main_of_sadtalker](https://user-images.githubusercontent.com/4397546/222490596-4c8a2115-49a7-42ad-a2c3-3bb3288a5f36.png) 


## 🚧 TODO

- [x] Generating 2D face from a single Image.
- [ ] Generating 3D face from Audio.
- [x] Generating 4D free-view talking examples from audio and a single image.
- [x] Gradio/Colab Demo.
- [ ] training code of each componments.
- [ ] Audio-driven Anime Avatar.
- [ ] interpolate ChatGPT for a conversation demo 🤔
- [ ] integrade with stable-diffusion-web-ui. (stay tunning!)

https://user-images.githubusercontent.com/4397546/222513483-89161f58-83d0-40e4-8e41-96c32b47bd4e.mp4


## 🔮 Inference Demo!

#### Dependence Installation

<details><summary>CLICK ME</summary>

```
git clone https://github.com/Winfredy/SadTalker.git
cd SadTalker 
conda create -n sadtalker python=3.8
source activate sadtalker
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
conda install ffmpeg
pip install ffmpy Cmake boost dlib-bin # [dlib-bin is much faster than dlib installation] conda install dlib 
pip install -r requirements.txt

### install gpfgan for enhancer
pip install git+https://github.com/TencentARC/GFPGAN

```  

</details>

#### Trained Models
<details><summary>CLICK ME</summary>

You can run the following script to put all the models in the right place.

```bash
bash utils/download_models.sh
```

OR download our pre-trained model from [google drive](https://drive.google.com/drive/folders/1Wd88VDoLhVzYsQ30_qDVluQr_Xm46yHT?usp=sharing) or our [github release page](https://github.com/Winfredy/SadTalker/releases/tag/v0.0.1), and then, put it in ./checkpoints.

| Model | Description
| :--- | :----------
|checkpoints/auido2exp_00300-model.pth | Pre-trained ExpNet in Sadtalker.
|checkpoints/auido2pose_00140-model.pth | Pre-trained PoseVAE in Sadtalker.
|checkpoints/mapping_00229-model.pth.tar | Pre-trained MappingNet in Sadtalker.
|checkpoints/facevid2vid_00189-model.pth.tar | Pre-trained face-vid2vid model from [the reappearance of face-vid2vid](https://github.com/zhanglonghao1992/One-Shot_Free-View_Neural_Talking_Head_Synthesis).
|checkpoints/epoch_20.pth | Pre-trained 3DMM extractor in [Deep3DFaceReconstruction](https://github.com/microsoft/Deep3DFaceReconstruction).
|checkpoints/wav2lip.pth | Highly accurate lip-sync model in [Wav2lip](https://github.com/Rudrabha/Wav2Lip).
|checkpoints/shape_predictor_68_face_landmarks.dat | Face landmark model used in [dilb](http://dlib.net/). 
|checkpoints/BFM | 3DMM library file.  
|checkpoints/hub | Face detection models used in [face alignment](https://github.com/1adrianb/face-alignment).

</details>

#### Generating 2D face from a single Image

```bash
python inference.py --driven_audio <audio.wav> \
                    --source_image <video.mp4 or picture.png> \
                    --batch_size <default equals 2, a larger run faster> \
                    --expression_scale <default is 1.0, a larger value will make the motion stronger> \
                    --result_dir <a file to store results> \
                    --enhancer <default is None, you can choose gfpgan or RestoreFormer>
```

<!-- ###### The effectness of enhancer `gfpgan`. -->
| basic        |  w/ gfpgan   | w/ gfpgan, w/ expression scale = 2  |
| ------------- |:-------------:| :-----:|
|  <video src="https://user-images.githubusercontent.com/4397546/226097707-bef1dd41-403e-48d3-a6e6-6adf923843af.mp4"></video>  | <video src="https://user-images.githubusercontent.com/4397546/226097717-12a1a2a1-ac0f-428d-b2cb-bd6917aff73e.mp4"></video>     | <video src="https://user-images.githubusercontent.com/4397546/226097732-fac6be37-cd87-44f0-a12a-be768337986f.mp4"></video>    |




#### Generating 3D face from Audio

To do ...

#### Generating 4D free-view talking examples from audio and a single image

We use `camera_yaw`, `camera_pitch`, `camera_roll` to control camera pose. For example, `--camera_yaw -20 30 10` means the camera yaw degree changes from -20 to 30 and then changes from 30 to 10.
```bash
python inference.py --driven_audio <audio.wav> \
                    --source_image <video.mp4 or picture.png> \
                    --result_dir <a file to store results> \
                    --camera_yaw -20 30 10
```
![free_view](https://github.com/Winfredy/SadTalker/blob/main/docs/free_view_result.gif)


## 🛎 Citation

If you find our work useful in your research, please consider citing:

```bibtex
@article{zhang2022sadtalker,
  title={SadTalker: Learning Realistic 3D Motion Coefficients for Stylized Audio-Driven Single Image Talking Face Animation},
  author={Zhang, Wenxuan and Cun, Xiaodong and Wang, Xuan and Zhang, Yong and Shen, Xi and Guo, Yu and Shan, Ying and Wang, Fei},
  journal={arXiv preprint arXiv:2211.12194},
  year={2022}
}
```

## 💗 Acknowledgements

Facerender code borrows heavily from [zhanglonghao's reproduction of face-vid2vid](https://github.com/zhanglonghao1992/One-Shot_Free-View_Neural_Talking_Head_Synthesis) and [PIRender](https://github.com/RenYurui/PIRender). We thank the authors for sharing their wonderful code. In training process, We also use the model from [Deep3DFaceReconstruction](https://github.com/microsoft/Deep3DFaceReconstruction) and [Wav2lip](https://github.com/Rudrabha/Wav2Lip). We thank for their wonderful work.


## 🥂 Related Works
- [StyleHEAT: One-Shot High-Resolution Editable Talking Face Generation via Pre-trained StyleGAN (ECCV 2022)](https://github.com/FeiiYin/StyleHEAT)
- [CodeTalker: Speech-Driven 3D Facial Animation with Discrete Motion Prior (CVPR 2023)](https://github.com/Doubiiu/CodeTalker)
- [VideoReTalking: Audio-based Lip Synchronization for Talking Head Video Editing In the Wild (SIGGRAPH Asia 2022)](https://github.com/vinthony/video-retalking)
- [DPE: Disentanglement of Pose and Expression for General Video Portrait Editing (CVPR 2023)](https://github.com/Carlyx/DPE)
- [3D GAN Inversion with Facial Symmetry Prior (CVPR 2023)](https://github.com/FeiiYin/SPI/)
- [T2M-GPT: Generating Human Motion from Textual Descriptions with Discrete Representations (CVPR 2023)](https://github.com/Mael-zys/T2M-GPT)

## 📢 Disclaimer

This is not an official product of Tencent. This repository can only be used for personal/research/non-commercial purposes.
