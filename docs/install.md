
### Mac (Tested on M1 Mac OS 13.3)

```
git clone https://github.com/Winfredy/SadTalker.git

cd SadTalker 

conda create -n sadtalker python=3.8

conda activate sadtalker

# install pytorch 2.0
pip install torch torchvision torchaudio

conda install ffmpeg

pip install -r requirements.txt

pip install dlib # mac need to install the original dlib.

```



### Windows Native

- Make sure you have `ffmpeg` in the `%PATH%` as suggested in [#54](https://github.com/Winfredy/SadTalker/issues/54), following [this](https://www.geeksforgeeks.org/how-to-install-ffmpeg-on-windows/) installation to install `ffmpeg`. 


### Windows WSL
- Make sure the environment: `export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH`


### Docker installnation

A dockerfile are also provided by [@thegenerativegeneration](https://github.com/thegenerativegeneration) in [docker hub](https://hub.docker.com/repository/docker/wawa9000/sadtalker), which can be used directly as:

```bash
docker run --gpus "all" --rm -v $(pwd):/host_dir wawa9000/sadtalker \
    --driven_audio /host_dir/deyu.wav \
    --source_image /host_dir/image.jpg \
    --expression_scale 1.0 \
    --still \
    --result_dir /host_dir
```

