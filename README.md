<div align="center">

<img src='https://user-images.githubusercontent.com/4397546/229094115-862c747e-7397-4b54-ba4a-bd368bfe2e0f.png' width='500px'/>


<!--<h2> 😭 SadTalker： <span style="font-size:12px">Learning Realistic 3D Motion Coefficients for Stylized Audio-Driven Single Image Talking Face Animation </span> </h2> -->

  <a href='https://arxiv.org/abs/2211.12194'><img src='https://img.shields.io/badge/ArXiv-PDF-red'></a> &nbsp; <a href='https://sadtalker.github.io'><img src='https://img.shields.io/badge/Project-Page-Green'></a> &nbsp; [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Winfredy/SadTalker/blob/main/quick_demo.ipynb) &nbsp; [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/vinthony/SadTalker) &nbsp; [![sd webui-colab](https://img.shields.io/badge/Automatic1111-Colab-green)](https://colab.research.google.com/github/camenduru/stable-diffusion-webui-colab/blob/main/video/stable/stable_diffusion_1_5_video_webui_colab.ipynb) &nbsp; [![Replicate](https://replicate.com/cjwbw/sadtalker/badge)](https://replicate.com/cjwbw/sadtalker) 

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



## 🔥 Highlight

- 🔥 The extension of the [stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui) is online. Checkout more details [here](docs/webui_extension.md).

https://user-images.githubusercontent.com/4397546/231495639-5d4bb925-ea64-4a36-a519-6389917dac29.mp4

- 🔥 `full image mode` is online! checkout [here](https://github.com/Winfredy/SadTalker#full-bodyimage-generation) for more details.

| still+enhancer in v0.0.1                 | still + enhancer   in v0.0.2       |   [input image @bagbag1815](https://twitter.com/bagbag1815/status/1642754319094108161) |
|:--------------------: |:--------------------: | :----: |
| <video  src="https://user-images.githubusercontent.com/48216707/229484996-5d7be64f-2553-4c9e-a452-c5cf0b8ebafe.mp4" type="video/mp4"> </video> | <video  src="https://user-images.githubusercontent.com/4397546/230717873-355b7bf3-d3de-49f9-a439-9220e623fce7.mp4" type="video/mp4"> </video>  | <img src='./examples/source_image/full_body_2.png' width='380'> 

- 🔥 Several new mode, eg, `still mode`, `reference mode`, `resize mode` are online for better and custom applications.

- 🔥 Happy to  see more community demos at [bilibili](https://search.bilibili.com/all?keyword=sadtalker&from_source=webtop_search&spm_id_from=333.1007&search_source=3
), [Youtube](https://www.youtube.com/results?search_query=sadtalker&sp=CAM%253D) and [twitter #sadtalker](https://twitter.com/search?q=%23sadtalker&src=typed_query).

## 📋 Changelog (Previous changelog can be founded [here](docs/changlelog.md))

- __[2023.06.05]__: release a new 512 beta face model. Fixed some bugs and improve the performance.

- __[2023.04.15]__: Adding automatic1111 colab by @camenduru, thanks for this awesome colab: [![sd webui-colab](https://img.shields.io/badge/Automatic1111-Colab-green)](https://colab.research.google.com/github/camenduru/stable-diffusion-webui-colab/blob/main/video/stable/stable_diffusion_1_5_video_webui_colab.ipynb).

- __[2023.04.12]__: adding a more detailed sd-webui installation document, fixed reinstallation problem.

- __[2023.04.12]__: Fixed the sd-webui safe issues becasue of the 3rd packages, optimize the output path in `sd-webui-extension`.

- __[2023.04.08]__: ❗️❗️❗️ In v0.0.2, we add a logo watermark to the generated video to prevent abusing since it is very realistic.

- __[2023.04.08]__: v0.0.2, full image animation, adding baidu driver for download checkpoints. Optimizing the logic about enhancer.


## 🚧 TODO

<details><summary> Previous TODOs </summary>

- [x] Generating 2D face from a single Image.
- [x] Generating 3D face from Audio.
- [x] Generating 4D free-view talking examples from audio and a single image.
- [x] Gradio/Colab Demo.
- [x] Full body/image Generation.
- [x] integrade with stable-diffusion-web-ui. (stay tunning!)
</details>


- [ ] Audio-driven Anime Avatar.
- [ ] training code of each componments.


## If you have any problem, please view our [FAQ](docs/FAQ.md) before opening an issue.

## ⚙️ 1. Installation.

Tutorials from communities: [中文windows教程](https://www.bilibili.com/video/BV1Dc411W7V6/) | [日本語コース](https://br-d.fanbox.cc/posts/5685086?utm_campaign=manage_post_page&utm_medium=share&utm_source=twitter) 

### Linux:

1. Installing [anaconda](https://www.anaconda.com/), python and git.

2. Creating the env and install the requirements.
  ```bash
  git clone https://github.com/Winfredy/SadTalker.git

  cd SadTalker 

  conda create -n sadtalker python=3.8

  conda activate sadtalker

  pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113

  conda install ffmpeg

  pip install -r requirements.txt

  ### tts is optional for gradio demo. 
  ### pip install TTS

  ```  
### Windows ([中文windows教程](https://www.bilibili.com/video/BV1Dc411W7V6/)):

1. Install [Python 3.10.6](https://www.python.org/downloads/windows/), checking "Add Python to PATH".
2. Install [git](https://git-scm.com/download/win) manually (OR `scoop install git` via [scoop](https://scoop.sh/)).
3. Install `ffmpeg`, following [this instruction](https://www.wikihow.com/Install-FFmpeg-on-Windows) (OR using `scoop install ffmpeg` via [scoop](https://scoop.sh/)).
4. Download our SadTalker repository, for example by running `git clone https://github.com/Winfredy/SadTalker.git`.
5. Download the `checkpoint` and `gfpgan` [below↓](https://github.com/Winfredy/SadTalker#-2-download-trained-models).
5. Run `start.bat` from Windows Explorer as normal, non-administrator, user, a gradio WebUI demo will be started.

### Macbook:

More tips about installnation on Macbook and the Docker file can be founded [here](docs/install.md)

## 📥 2. Download Trained Models.

You can run the following script to put all the models in the right place.

```bash
bash scripts/download_models.sh
```

Other alternatives:
> we also provide an offline patch (`gfpgan/`), thus, no model will be downloaded when generating.

**Google Driver**: download our pre-trained model from [ this link (main checkpoints)](https://drive.google.com/file/d/1gwWh45pF7aelNP_P78uDJL8Sycep-K7j/view?usp=sharing) and [ gfpgan (offline patch)](https://drive.google.com/file/d/19AIBsmfcHW6BRJmeqSFlG5fL445Xmsyi?usp=sharing)

**Github Release Page**: download all the files from the [lastest github release page](https://github.com/Winfredy/SadTalker/releases), and then, put it in ./checkpoints.

**百度云盘**: we provided the downloaded model in [checkpoints,  提取码: sadt.](https://pan.baidu.com/s/1P4fRgk9gaSutZnn8YW034Q?pwd=sadt) And [gfpgan,  提取码: sadt.](https://pan.baidu.com/s/1kb1BCPaLOWX1JJb9Czbn6w?pwd=sadt)



<details><summary>Model Details</summary>


Model explains:

##### New version 
| Model | Description
| :--- | :----------
|checkpoints/mapping_00229-model.pth.tar | Pre-trained MappingNet in Sadtalker.
|checkpoints/mapping_00109-model.pth.tar | Pre-trained MappingNet in Sadtalker.
|checkpoints/SadTalker_V0.0.2_256.safetensors | packaged sadtalker checkpoints of old version, 256 face render).
|checkpoints/SadTalker_V0.0.2_512.safetensors | packaged sadtalker checkpoints of old version, 512 face render).
|gfpgan/weights | Face detection and enhanced models used in `facexlib` and `gfpgan`.
  
  
##### Old version
| Model | Description
| :--- | :----------
|checkpoints/auido2exp_00300-model.pth | Pre-trained ExpNet in Sadtalker.
|checkpoints/auido2pose_00140-model.pth | Pre-trained PoseVAE in Sadtalker.
|checkpoints/mapping_00229-model.pth.tar | Pre-trained MappingNet in Sadtalker.
|checkpoints/mapping_00109-model.pth.tar | Pre-trained MappingNet in Sadtalker.
|checkpoints/facevid2vid_00189-model.pth.tar | Pre-trained face-vid2vid model from [the reappearance of face-vid2vid](https://github.com/zhanglonghao1992/One-Shot_Free-View_Neural_Talking_Head_Synthesis).
|checkpoints/epoch_20.pth | Pre-trained 3DMM extractor in [Deep3DFaceReconstruction](https://github.com/microsoft/Deep3DFaceReconstruction).
|checkpoints/wav2lip.pth | Highly accurate lip-sync model in [Wav2lip](https://github.com/Rudrabha/Wav2Lip).
|checkpoints/shape_predictor_68_face_landmarks.dat | Face landmark model used in [dilb](http://dlib.net/). 
|checkpoints/BFM | 3DMM library file.  
|checkpoints/hub | Face detection models used in [face alignment](https://github.com/1adrianb/face-alignment).
|gfpgan/weights | Face detection and enhanced models used in `facexlib` and `gfpgan`.

The final folder will be shown as:

<img width="331" alt="image" src="https://user-images.githubusercontent.com/4397546/232511411-4ca75cbf-a434-48c5-9ae0-9009e8316484.png">


</details>

## 🔮 3. Quick Start ([Best Practice](docs/best_practice.md)).

### WebUI Demos:

**Online**: [Huggingface](https://huggingface.co/spaces/vinthony/SadTalker) | [SDWebUI-Colab](https://colab.research.google.com/github/camenduru/stable-diffusion-webui-colab/blob/main/video/stable/stable_diffusion_1_5_video_webui_colab.ipynb) | [Colab](https://colab.research.google.com/github/Winfredy/SadTalker/blob/main/quick_demo.ipynb)

**Local Autiomatic1111 stable-diffusion webui extension**: please refer to [Autiomatic1111 stable-diffusion webui docs](docs/webui_extension.md).

**Local gradio demo**: Similar to our [hugging-face demo](https://huggingface.co/spaces/vinthony/SadTalker) can be run by:

```bash
## you need manually install TTS(https://github.com/coqui-ai/TTS) via `pip install tts` in advanced.
python app.py
```

**Local windows gradio demo**: just double click `webui.bat`, the requirements will be installed automatically.


### Manually usages:

##### Animating a portrait image from default config:
```bash
python inference.py --driven_audio <audio.wav> \
                    --source_image <video.mp4 or picture.png> \
                    --enhancer gfpgan 
```
The results will be saved in `results/$SOME_TIMESTAMP/*.mp4`.

##### Full body/image Generation:

Using `--still` to generate a natural full body video. You can add `enhancer` to improve the quality of the generated video. 

```bash
python inference.py --driven_audio <audio.wav> \
                    --source_image <video.mp4 or picture.png> \
                    --result_dir <a file to store results> \
                    --still \
                    --preprocess full \
                    --enhancer gfpgan 
```

More examples and configuration and tips can be founded in the [ >>> best practice documents <<<](docs/best_practice.md).

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

See also these wonderful 3rd libraries we use:

- **Face Utils**: https://github.com/xinntao/facexlib
- **Face Enhancement**: https://github.com/TencentARC/GFPGAN
- **Image/Video Enhancement**:https://github.com/xinntao/Real-ESRGAN

## 🥂 Extensions:

- [SadTalker-Video-Lip-Sync](https://github.com/Zz-ww/SadTalker-Video-Lip-Sync) from [@Zz-ww](https://github.com/Zz-ww): SadTalker for Video Lip Editing

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

All the copyright of the demo images and audio are from communities users or the geneartion from stable diffusion. Free free to contact us if you feel uncomfortable.

