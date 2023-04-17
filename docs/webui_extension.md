
## Run SadTalker as a Stable Diffusion WebUI Extension.

1. Installing the lastest version of [stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui) and install the sadtalker via `extension`.
<img width="726" alt="image" src="https://user-images.githubusercontent.com/4397546/230698519-267d1d1f-6e99-4dd4-81e1-7b889259efbd.png">

2. Download the checkpoints manually, for Linux and Mac:

    ```bash

    cd SOMEWHERE_YOU_LIKE

    bash <(wget -qO- https://raw.githubusercontent.com/Winfredy/SadTalker/main/scripts/download_models.sh)
    ```

    For windows, you can download all the checkpoints from [google drive](https://drive.google.com/drive/folders/1Wd88VDoLhVzYsQ30_qDVluQr_Xm46yHT?usp=sharing) or [百度云盘](https://pan.baidu.com/s/1nXuVNd0exUl37ISwWqbFGA?pwd=sadt) 提取码: sadt.

3.1. options 1: put the checkpoint in `stable-diffusion-webui/models/SadTalker` or `stable-diffusion-webui/extensions/SadTalker/checkpoints/`, the checkpoints will be detected automatically.

3.2. Options 2: Set the path of `SADTALKTER_CHECKPOINTS` in `webui_user.sh`(linux) or `webui_user.bat`(windows) by:

    > only works if you are directly starting webui from `webui_user.sh` or `webui_user.bat`.

    ```bash
    # windows (webui_user.bat)
    set SADTALKER_CHECKPOINTS=D:\SadTalker\checkpoints

    # linux (webui_user.sh)
    export SADTALKER_CHECKPOINTS=/path/to/SadTalker/checkpoints
    ```

4. Then, starting the webui via `webui.sh or webui_user.sh(linux)` or `webui_user.bat(windows)` or any other methods, the SadTalker can be used in stable-diffusion-webui directly. 
    
    <img width="726" alt="image" src="https://user-images.githubusercontent.com/4397546/230698614-58015182-2916-4240-b324-e69022ef75b3.png">
    
## Questsions

1. if you are running on CPU, you need to specific `--disable-safe-unpickle` in `webui_user.sh` or `webui_user.bat`.

    ```bash
    # windows (webui_user.bat)
    set COMMANDLINE_ARGS="--disable-safe-unpickle"

    # linux (webui_user.sh)
    export COMMANDLINE_ARGS="--disable-safe-unpickle"
    ```



(Some [important discussion](https://github.com/Winfredy/SadTalker/issues/78) if you are unable to use `full` mode).
