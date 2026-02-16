import pandas as pd
import numpy as np
import os
import shutil
from tqdm.auto import tqdm

source_dataset = "BRB_Dataset_Test"
target_dataset = "BRB_Dataset"

df = pd.read_csv("zipfile/Train.csv")
for camera in os.listdir(source_dataset):
    for video in tqdm(os.listdir(os.path.join(source_dataset, camera))):
        video_name = f"{camera}/{video}"
        if video_name in df["videos"].values:
            source_video_path = os.path.join(source_dataset, camera, video)
            target_video_path = os.path.join(target_dataset, camera, video)
            shutil.move(source_video_path, target_video_path)
