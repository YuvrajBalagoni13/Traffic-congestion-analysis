import argparse
import numpy as np
import pandas as pd
import json
import os
from tqdm.auto import tqdm

from core.videoprocessor import VideoProcessor

dataset_path = "BRB_Dataset"
cameras = os.listdir(dataset_path)

df = pd.read_csv("dataset_features_yolo26.csv")
# df = pd.DataFrame()

train_data = pd.read_csv("zipfile/Train.csv")

'''
this code is trying to see if 5 mins in the future is free flowing then considering it to be skipped.
'''
mask = (train_data['congestion_enter_rating'] == "free flowing")
for i in range(1, 5):
    mask &= (train_data['congestion_enter_rating'].shift(-i) == "free flowing")

'''
Doing this to perform undersampling before the processing to reduce time & resource.
counting the number of videos we are going to skip & creating a list of videos to skip (got around 1752 videos from 4640)
'''
camera_count = 4020
data_count = 1160
segments = [sum(mask.iloc[i * camera_count : i * camera_count + data_count]) for i in range(4)]
total_videos_skipped = sum(segments)

video_list_ff = train_data.loc[mask, 'videos'].sample(frac=0.6, random_state=42).str.split("/").str[-1].tolist()

print("=" * 40)
print(f"skipping total of {total_videos_skipped} videos for undersampling reasons\n")
print("=" * 40)

processor = VideoProcessor(
    source_weights_path = "runs/detect/yolo26_custom_run/weights/best.pt",
    confidence_threshold = 0.4
)

feature_output_dict = {}
rows = []
for camera in cameras:
    if camera != "normanniles4":
        continue
    camera_path = f"{dataset_path}/{camera}"
    video_list = os.listdir(camera_path)
    print(len(video_list))
    for idx, video in tqdm(enumerate(video_list)):
        '''
        Skipping videos from 6 to 10 AM as it is mostly free flowing (Another measure to reduce the imbalance)
        '''
        if idx < 200:
            continue

        source_video_path = f"{camera_path}/{video}"
        if not df.empty:
            if video in df["video_name"].values or video in video_list_ff:
                continue

        print(source_video_path)

        if not os.path.exists(source_video_path):
            print(f"Skipping: {source_video_path} does not exist!")
            continue

        feature_output = processor.process_video(source_video_path = source_video_path, frame_skip=1)
        feature_row = processor.create_feature_row()

        if feature_output is None:
            continue

        new_row = pd.DataFrame([feature_row])
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv("dataset_features_yolo26.csv", index=False)

        feature_output_dict[video] = feature_output
        rows.append(feature_row)

with open("BRB_dataset_features.json", "w") as f:
    json.dump(feature_output_dict, f, indent=4)

dataframe = pd.DataFrame(rows)
dataframe.to_csv("updated_dataset_features.csv", index=True)

"""
python -m utils.create_tab_dataset
"""