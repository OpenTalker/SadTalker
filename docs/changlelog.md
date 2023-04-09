## changelogs


- __[2023.04.06]__: stable-diffiusion webui extension is release.

- __[2023.04.03]__: Enable TTS in huggingface and gradio local demo.

- __[2023.03.30]__: Launch beta version of the full body mode.

- __[2023.03.30]__: Launch new feature: through using reference videos, our algorithm can generate videos with more natural eye blinking and some eyebrow movement.

- __[2023.03.29]__: `resize mode` is online by `python infererence.py --preprocess resize`! Where we can produce a larger crop of the image as discussed in https://github.com/Winfredy/SadTalker/issues/35.

- __[2023.03.29]__: local gradio demo is online! `python app.py` to start the demo. New `requirments.txt` is used to avoid the bugs in `librosa`.

- __[2023.03.28]__: Online demo is launched in [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/vinthony/SadTalker), thanks AK!
 
- __[2023.03.22]__: Launch new feature: generating the 3d face animation from a single image. New applications about it will be updated.

- __[2023.03.22]__: Launch new feature: `still mode`, where only a small head pose will be produced via `python inference.py --still`. 

- __[2023.03.18]__: Support `expression intensity`, now you can change the intensity of the generated motion: `python inference.py --expression_scale 1.3 (some value > 1)`.

- __[2023.03.18]__: Reconfig the data folders, now you can download the checkpoint automatically using `bash scripts/download_models.sh`.
- __[2023.03.18]__: We have offically integrate the [GFPGAN](https://github.com/TencentARC/GFPGAN) for face enhancement, using `python inference.py --enhancer gfpgan` for  better visualization performance.
- __[2023.03.14]__: Specify the version of package `joblib` to remove the errors in using `librosa`, [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Winfredy/SadTalker/blob/main/quick_demo.ipynb) is online!
- __[2023.03.06]__: Solve some bugs in code and errors in installation 
- __[2023.03.03]__: Release the test code for audio-driven single image animation!
- __[2023.02.28]__: SadTalker has been accepted by CVPR 2023!
