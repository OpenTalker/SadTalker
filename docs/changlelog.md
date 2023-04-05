## changelogs

  - __[2023.03.22]__: Launch new feature: generating the 3d face animation from a single image. New applications about it will be updated.

  - __[2023.03.22]__: Launch new feature: `still mode`, where only a small head pose will be produced via `python inference.py --still`. 

  - __[2023.03.18]__: Support `expression intensity`, now you can change the intensity of the generated motion: `python inference.py --expression_scale 1.3 (some value > 1)`.

  - __[2023.03.18]__: Reconfig the data folders, now you can download the checkpoint automatically using `bash scripts/download_models.sh`.
  - __[2023.03.18]__: We have offically integrate the [GFPGAN](https://github.com/TencentARC/GFPGAN) for face enhancement, using `python inference.py --enhancer gfpgan` for  better visualization performance.
  - __[2023.03.14]__: Specify the version of package `joblib` to remove the errors in using `librosa`, [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Winfredy/SadTalker/blob/main/quick_demo.ipynb) is online!
  - __[2023.03.06]__: Solve some bugs in code and errors in installation 
  - __[2023.03.03]__: Release the test code for audio-driven single image animation!
  - __[2023.02.28]__: SadTalker has been accepted by CVPR 2023!
