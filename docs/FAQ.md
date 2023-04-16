
## Frequency Asked Question

**Q: `ffmpeg` is not recognized as an internal or external command**

In Linux, you can install the ffmpeg via `conda install ffmpeg`. Or on Mac OS X, try to install ffmpeg via `brew install ffmpeg`. On windows, make sure you have `ffmpeg` in the `%PATH%` as suggested in [#54](https://github.com/Winfredy/SadTalker/issues/54), then, following [this](https://www.geeksforgeeks.org/how-to-install-ffmpeg-on-windows/) installation to install `ffmpeg`.

**Q: Running Requirments.**

Please refer to the discussion here: https://github.com/Winfredy/SadTalker/issues/124#issuecomment-1508113989


**Q: ModuleNotFoundError: No module named 'ai'**

please check the checkpoint's size of the `epoch_20.pth`. (https://github.com/Winfredy/SadTalker/issues/167, https://github.com/Winfredy/SadTalker/issues/113)

**Q: Illegal Hardware Error: Mac M1**

please reinstall the `dlib` by `pip install dlib` individually. (https://github.com/Winfredy/SadTalker/issues/129, https://github.com/Winfredy/SadTalker/issues/109)
