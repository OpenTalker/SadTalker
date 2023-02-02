
<div align="center">

<h2> 😭 SadTalker： <span style="font-size:12px">Learning Realistic 3D Motion Coefficients for  Stylized Audio-Driven </br> Single Image Talking Face Animation</span> </h2> 

  <a href='https://arxiv.org/abs/2211.12194'><img src='https://img.shields.io/badge/ArXiv-2211.14758-red'></a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href='https://sadtalker.github.io'><img src='https://img.shields.io/badge/Project-Page-Green'></a>

<div>
    <a target='_blank'>Wenxuan Zhang <sup>*,1,2</sup> </a>&emsp;
    <a href='https://vinthony.github.io/' target='_blank'>Xiaodong Cun <sup>*,2</a>&emsp;
    <a href='https://xuanwangvc.github.io/' target='_blank'>Xuan Wang <sup>2</sup></a>&emsp;
    <a href='https://yzhang2016.github.io/yongnorriszhang.github.io/' target='_blank'>Yong Zhang <sup>2</sup></a>&emsp;
    <a href='https://xishen0220.github.io/' target='_blank'>Xi Shen <sup>2</sup></a>&emsp;
    <a target='_blank'>Guo Yu <sup>1</sup> </a>&emsp;
    <a target='_blank'>Ying Shan <sup>2</sup> </a>&emsp;
    <a target='_blank'>Fei Wang <sup>1</sup> </a>&emsp;
</div>
<br>
<div>
    <sup>1</sup> Xi'an Jiaotong University &emsp; <sup>2</sup> Tencent AI Lab &emsp; 
</div>
<br>
<i><strong><a href='https://arxiv.org/abs/2211.12194' target='_blank'>Arxiv</a></strong></i>
<br>
<br>

<div align="justify"> Generating talking head videos through a face image and a piece of speech audio still contains many challenges. ie, unnatural head movement, distorted expression, and identity modification. We argue that these issues are mainly because of learning from the coupled 2D motion fields. On the other hand, explicitly using 3D information also suffers problems of stiff expression and incoherent video. We present SadTalker, which generates 3D motion coefficients (head pose, expression) of the 3DMM from audio and implicitly modulates a novel 3D-aware face render for talking head generation. To learn the realistic motion coefficients, we explicitly model the connections between audio and different types of motion coefficients individually. Precisely, we present ExpNet to learn the accurate facial expression from audio by distilling both coefficients and 3D-rendered faces. As for the head pose, we design PoseVAE via a conditional VAE to synthesize head motion in different styles. Finally, the generated 3D motion coefficients are mapped to the unsupervised 3D keypoints space of the proposed face render, and synthesize the final video. We conduct extensive experiments to show the superior of our method in terms of motion and video quality.</div>
<br>

</div>

## **TODO**

- [ ] Generating 2D face from a single Image.
- [ ] Generating 3D face from Audio.
- [ ] Generating 4D free-view talking examples from audio and a single image.
- [ ] Gradio/Colab Demo.
- [ ] integrade with stable-diffusion-web-ui.
- [ ] training code of each componments.

## **Citation**

If you find our work useful in your research, please consider citing:

```
@article{zhang2022sadtalker,
  title={SadTalker: Learning Realistic 3D Motion Coefficients for Stylized Audio-Driven Single Image Talking Face Animation},
  author={Zhang, Wenxuan and Cun, Xiaodong and Wang, Xuan and Zhang, Yong and Shen, Xi and Guo, Yu and Shan, Ying and Wang, Fei},
  journal={arXiv preprint arXiv:2211.12194},
  year={2022}
}
```

