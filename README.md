<div align="center">

<img src='https://user-images.githubusercontent.com/4397546/229094115-862c747e-7397-4b54-ba4a-bd368bfe2e0f.png' width='500px'/>


<!--<h2> 😭 SadTalker： <span style="font-size:12px">Learning Realistic 3D Motion Coefficients for  Stylized Audio-Driven Single Image Talking Face Animation </span> </h2> -->

  <a href='https://arxiv.org/abs/2211.12194'><img src='https://img.shields.io/badge/ArXiv-PDF-red'></a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href='https://sadtalker.github.io'><img src='https://img.shields.io/badge/Project-Page-Green'></a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Winfredy/SadTalker/blob/main/quick_demo.ipynb) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/vinthony/SadTalker)


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

<b>TL;DR: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; single portrait image 🙎‍♂️  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; audio 🎤  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; =  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; talking head video 🎞.</b>

<br>

</div>

## 🔥🔥🔥 Highlight
- Several new mode, eg, `still mode`, `reference mode`, `resize mode` are online for better and custom applications.
- Happy to see our method is used in various talking or singing avatar, checkout these wonderful demos at [bilibili](https://search.bilibili.com/all?keyword=sadtalker&from_source=webtop_search&spm_id_from=333.1007&search_source=3
) and [twitter #sadtalker](https://twitter.com/search?q=%23sadtalker&src=typed_query).

## 📋 Changelog

- __[2023.03.30]__: Launch new feature: through using reference videos, our algorithm can generate videos with more natural eye blinking and some eyebrow movement.

- __[2023.03.29]__: `resize mode` is online by `python infererence.py --preprocess resize`! Where we can produce a larger crop of the image as discussed in https://github.com/Winfredy/SadTalker/issues/35.

- __[2023.03.29]__: local gradio demo is online! `python app.py` to start the demo. New `requirments.txt` is used to avoid the bugs in `librosa`.

- __[2023.03.28]__: Online demo is launched in [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/vinthony/SadTalker), thanks AK!

- __[2023.03.22]__: Launch new feature: generating the 3d face animation from a single image. New applications about it will be updated.

- __[2023.03.22]__: Launch new feature: `still mode`, where only a small head pose will be produced via `python inference.py --still`. 

&nbsp;&nbsp;&nbsp;&nbsp; <details><summary> Previous Changelogs</summary>

  - __[2023.03.18]__: Support `expression intensity`, now you can change the intensity of the generated motion: `python inference.py --expression_scale 1.3 (some value > 1)`.

  - __[2023.03.18]__: Reconfig the data folders, now you can download the checkpoint automatically using `bash scripts/download_models.sh`.
  - __[2023.03.18]__: We have offically integrate the [GFPGAN](https://github.com/TencentARC/GFPGAN) for face enhancement, using `python inference.py --enhancer gfpgan` for  better visualization performance.
  - __[2023.03.14]__: Specify the version of package `joblib` to remove the errors in using `librosa`, [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Winfredy/SadTalker/blob/main/quick_demo.ipynb) is online!
  - __[2023.03.06]__: Solve some bugs in code and errors in installation 
  - __[2023.03.03]__: Release the test code for audio-driven single image animation!
  - __[2023.02.28]__: SadTalker has been accepted by CVPR 2023!

</details>

## 🎼 Pipeline
![main_of_sadtalker](https://user-images.githubusercontent.com/4397546/222490596-4c8a2115-49a7-42ad-a2c3-3bb3288a5f36.png) 


## 🚧 TODO

- [x] Generating 2D face from a single Image.
- [x] Generating 3D face from Audio.
- [x] Generating 4D free-view talking examples from audio and a single image.
- [x] Gradio/Colab Demo.
- [ ] Full body/image Generation.
- [ ] training code of each componments.
- [ ] Audio-driven Anime Avatar.
- [ ] interpolate ChatGPT for a conversation demo 🤔
- [ ] integrade with stable-diffusion-web-ui. (stay tunning!)

https://user-images.githubusercontent.com/4397546/222513483-89161f58-83d0-40e4-8e41-96c32b47bd4e.mp4


## 🔮 Installation

#### Dependence Installation

<details><summary>CLICK ME For Mannual Installation </summary>

```
git clone https://github.com/Winfredy/SadTalker.git
cd SadTalker 
conda create -n sadtalker python=3.8
source activate sadtalker
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
conda install ffmpeg
pip install dlib-bin # [dlib-bin is much faster than dlib installation] conda install dlib 
pip install -r requirements.txt

### install gpfgan for enhancer
pip install git+https://github.com/TencentARC/GFPGAN

```  

</details>

<details><summary>CLICK For Docker Installation </summary>

A dockerfile are also provided by [@thegenerativegeneration](https://github.com/thegenerativegeneration) in [docker hub](https://hub.docker.com/repository/docker/wawa9000/sadtalker), which can be used directly as:

```bash
docker run --gpus "all" --rm -v $(pwd):/host_dir wawa9000/sadtalker \
    --driven_audio /host_dir/deyu.wav \
    --source_image /host_dir/image.jpg \
    --expression_scale 1.0 \
    --still \
    --result_dir /host_dir
```
</details>


#### Trained Models
<details><summary>CLICK ME</summary>

You can run the following script to put all the models in the right place.

```bash
bash scripts/download_models.sh
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

## 🔮 Inference Demo

#### Generating 2D face from a single Image

```bash
python inference.py --driven_audio <audio.wav> \
                    --source_image <video.mp4 or picture.png> \
                    --batch_size <default equals 2, a larger run faster> \
                    --expression_scale <default is 1.0, a larger value will make the motion stronger> \
                    --result_dir <a file to store results> \
                    --still <add this flag will show fewer head motion> \
                    --preprocess <resize or crop the input image, default is crop> \
                    --enhancer <default is None, you can choose gfpgan or RestoreFormer> \
                    --full_img_enhancer <default is None, you can choose gfpgan or RestoreFormer> \
                    --ref_eyeblink <default is None, ref_eyeblink is used to provide more natural eyebrow movement and eye blinking> \ 
                    --ref_pose <default is None, ref_pose is used to provide head pose> 

```

<!-- ###### The effectness of enhancer `gfpgan`. -->

| basic        | w/ still mode |  w/ exp_scale 1.3   | w/ gfpgan  |
|:-------------: |:-------------: |:-------------: |:-------------: |
|  <video src="https://user-images.githubusercontent.com/4397546/226097707-bef1dd41-403e-48d3-a6e6-6adf923843af.mp4"></video>  | <video src='https://user-images.githubusercontent.com/4397546/226804933-b717229f-1919-4bd5-b6af-bea7ab66cad3.mp4'></video>  |  <video style='width:256px' src="https://user-images.githubusercontent.com/4397546/226806013-7752c308-8235-4e7a-9465-72d8fc1aa03d.mp4"></video>     | <video style='width:256px' src="https://user-images.githubusercontent.com/4397546/226097717-12a1a2a1-ac0f-428d-b2cb-bd6917aff73e.mp4"></video>    |
> Kindly ensure to activate the audio as the default audio playing is incompatible with GitHub.

| Input, w/ reference video   ,  reference video    | 
|:-------------: | 
|  ![free_view](docs/using_ref_video.gif)| 
| If the reference video is shorter than the input audio, we will loop the reference video . 



<!-- <video src="./docs/art_0##japanese_still.mp4"></video> -->


#### Generating 3D face from Audio


| Input        | Animated 3d face | 
|:-------------: | :-------------: |
|  <img src='examples/source_image/art_0.png' width='200px'> | <video src="https://user-images.githubusercontent.com/4397546/226856847-5a6a0a4d-a5ec-49e2-9b05-3206db65e8e3.mp4"></video>  | 

> Kindly ensure to activate the audio as the default audio playing is incompatible with GitHub.

More details to generate the 3d face can be founded [here](docs/face3d.md)


#### Generating 4D free-view talking examples from audio and a single image

We use `input_yaw`, `input_pitch`, `input_roll` to control head pose. For example, `--input_yaw -20 30 10` means the input head yaw degree changes from -20 to 30 and then changes from 30 to 10.
```bash
python inference.py --driven_audio <audio.wav> \
                    --source_image <video.mp4 or picture.png> \
                    --result_dir <a file to store results> \
                    --input_yaw -20 30 10
```
![free_view](docs/free_view_result.gif)

#### Full body/image Generation

Now you can use `--still` to generate a natural full body video. You can add `enhancer` or `full_img_enhancer` to improve the quality of the generated video. However, if you add other mode, such as `ref_eyeblinking`, `ref_pose`, the result will be bad. We are still trying to fix this problem.

```bash
python inference.py --driven_audio <audio.wav> \
                    --source_image <video.mp4 or picture.png> \
                    --result_dir <a file to store results> \
                    --still \
                    --enhancer gfpgan \
                    --full_img_enhancer gfpgan
```

| still                 | still + enhancer          |  still+full_img_enhancer  |
|:--------------------: |:--------------------: |:-----------------------:|
|  <video src="https://github.com/Winfredy/SadTalker/blob/main/docs/still1.mp4"></video>  | <video src=" https://github.com/Winfredy/SadTalker/blob/main/docs/still_e.mp4"></video>   | <video src="https://github.com/Winfredy/SadTalker/blob/main/docs/still1.mp4"></video>    | 







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

LOGO: color and font suggestion: [ChatGPT](ai.com), logo font：[Montserrat Alternates
](https://fonts.google.com/specimen/Montserrat+Alternates?preview.text=SadTalker&preview.text_type=custom&query=mont).
