import requests
import os
import zipfile

from tqdm import tqdm

checkpoints_dir = "checkpoints"
os.makedirs(checkpoints_dir, exist_ok=True)

gfpgan_dir = os.path.join("gfpgan", "weights")
os.makedirs(gfpgan_dir, exist_ok=True)


files_to_download = [
    (
        "https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2/auido2exp_00300-model.pth",
        "auido2exp_00300-model.pth",
        checkpoints_dir,
    ),
    (
        "https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2/auido2pose_00140-model.pth",
        "auido2pose_00140-model.pth",
        checkpoints_dir,
    ),
    (
        "https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2/BFM_Fitting.zip",
        "BFM_Fitting.zip",
        checkpoints_dir,
    ),
    (
        "https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2/epoch_20.pth",
        "epoch_20.pth",
        checkpoints_dir,
    ),
    (
        "https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2/facevid2vid_00189-model.pth.tar",
        "facevid2vid_00189-model.pth.tar",
        checkpoints_dir,
    ),
    (
        "https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2/hub.zip",
        "hub.zip",
        checkpoints_dir,
    ),
    (
        "https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2/mapping_00229-model.pth.tar",
        "mapping_00229-model.pth.tar",
        checkpoints_dir,
    ),
    (
        "https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2/shape_predictor_68_face_landmarks.dat",
        "shape_predictor_68_face_landmarks.dat",
        checkpoints_dir,
    ),
    (
        "https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2/wav2lip.pth",
        "wav2lip.pth",
        checkpoints_dir,
    ),
    (
        "https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2/mapping_00109-model.pth.tar",
        "mapping_00109-model.pth.tar",
        checkpoints_dir,
    ),
    (
        "https://github.com/xinntao/facexlib/releases/download/v0.1.0/alignment_WFLW_4HG.pth",
        "alignment_WFLW_4HG.pth",
        gfpgan_dir,
    ),
    (
        "https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth",
        "detection_Resnet50_Final.pth",
        gfpgan_dir,
    ),
    (
        "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth",
        "GFPGANv1.4.pth",
        gfpgan_dir,
    ),
    (
        "https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth",
        "parsing_parsenet.pth",
        gfpgan_dir,
    ),
]


def download_file(url, file_name, dir):
    response = requests.get(url, stream=True)
    total_size_in_bytes = int(response.headers.get("content-length", 0))
    block_size = 1024  # 1 Kibibyte
    progress_bar = tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True)
    progress_bar.set_description("Downloading: " + file_name)

    with open(os.path.join(dir, file_name), "wb") as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)

    progress_bar.close()
    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        raise Exception("Download failed: " + file_name)


def extract_zip(file_name, dir):
    zip_path = os.path.join(dir, file_name)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        file_names = zip_ref.namelist()
        total_size = sum([info.file_size for info in zip_ref.infolist()])
        progress_bar = tqdm(total=total_size, unit="B", unit_scale=True)
        progress_bar.set_description("Extracting: " + file_name)

        for file_name in file_names:
            zip_ref.extract(member=file_name, path=dir)
            progress_bar.update(zip_ref.getinfo(file_name).file_size)

        progress_bar.close()

    os.remove(zip_path)


progress_bar = tqdm(total=len(files_to_download), unit="files")
progress_bar.set_description("Overall Progress")

for url, file_name, dir in files_to_download:
    download_file(url, file_name, dir)

    if file_name.endswith(".zip"):
        extract_zip(file_name, dir)

    progress_bar.update()

progress_bar.close()
print("Downloaded all weights.")
