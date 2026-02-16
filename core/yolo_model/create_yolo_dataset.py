import cv2
import argparse
import numpy as np
import pandas as pd
import supervision as sv
import os
from tqdm.auto import tqdm

from videoprocessor import VideoProcessor

SOURCE_VIDEO_PATH = "BRB_Dataset"

class CreateYOLODataset:
    def __init__(
            self,
            source_weight_path: str,
            source_video_path: str,
            destination_path: str,
            video_skip: int = 1,
            frame_skip: int = 30,
            confidence_threshold: int = 0.4,
            iou_threshold: int = 0.3
    ) -> None:
        self.source_weight_path = source_weight_path
        self.source_video_path = source_video_path
        self.destination_path = destination_path
        self.video_skip = video_skip
        self.frame_skip = frame_skip
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.data_count = 0
        
        os.makedirs(self.destination_path, exist_ok=True)

        self.train_path = os.path.join(self.destination_path, "train")
        os.makedirs(self.train_path, exist_ok=True)

        self.image_path = os.path.join(self.train_path, "images")
        self.label_path = os.path.join(self.train_path, "labels")
        os.makedirs(self.image_path, exist_ok=True)
        os.makedirs(self.label_path, exist_ok=True)

        self.video_processor = VideoProcessor(
            source_weights_path = self.source_weight_path,
            confidence_threshold = self.confidence_threshold,
            iou_threshold = self.iou_threshold
        )
    
    def __len__(self) -> int:
        return self.data_count
    
    def process_detections(
            self,
            detections: sv.Detections
    ) -> str:
        detections_list = []
        
        for detection in detections:
            x, y, x_w, y_h = detection[0] # xyxy
            width, height = (x_w - x) / self.frame_width, (y_h - y) / self.frame_height
            x_center, y_center = (x + x_w) / (2 * self.frame_width), (y + y_h) / (2 * self.frame_height)
            class_detected = int(detection[3]) # class_id
            current_annotation = f"{class_detected} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
            detections_list.append(current_annotation)

        annotation = "\n".join(detections_list)
        return annotation  
    
    def process(self) -> None:
        folders = os.listdir(self.source_video_path)

        for folder in folders:
            print(f"processing - {folder}")
            videos = os.listdir(os.path.join(self.source_video_path, folder))
            for i, video in enumerate(videos):
                if i % (self.video_skip + 1) != 0:
                    continue
                video_path = os.path.join(self.source_video_path, folder, video)
                print(f"------------------- processing video - {video_path} -------------------")
                
                video = cv2.VideoCapture(video_path)
                self.frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
                self.frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
                self.total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                self.fps = video.get(cv2.CAP_PROP_FPS) if video.get(cv2.CAP_PROP_FPS) > 0 else 20.0
                print(f"fps - {self.fps}")

                pbar = tqdm(total=self.total_frames, desc="Processing Video")
                frame_count = 1

                if not video.isOpened():
                    print("Video not found")
                    continue
                
                while video.isOpened():
                    ret, frame = video.read()
                    
                    if not ret or frame is None:
                        print("End of video stream or failed to read frame.")
                        break

                    if frame_count % (self.frame_skip + 1) == 0:

                        if os.path.exists(os.path.join(self.label_path, f"{self.data_count:06d}.txt")):
                            self.data_count += 1
                            continue

                        image = frame.copy()

                        detections = self.video_processor.process_frame(image, track=False)
                        annotation = self.process_detections(detections)

                        with open(os.path.join(self.label_path, f"{self.data_count:06d}.txt"), "w") as f:
                            f.write(annotation)

                        cv2.imwrite(os.path.join(self.image_path, f"{self.data_count:06d}.jpg"), image)

                        self.data_count += 1
                    pbar.update(1)
                    frame_count += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="creating dataset for YOLO training"
    )

    parser.add_argument(
        "--source_weight_path",
        required = True,
        help = "path to models weights",
        type = str,
    )
    parser.add_argument(
        "--source_video_path",
        required = True,
        default = "BRB_Dataset",
        help = "video datasets path",
        type = str,
    )
    parser.add_argument(
        "--destination_path",
        required = True,
        help = "path to save new dataset",
        type = str,
    )
    parser.add_argument(
        "--video_skip",
        default = 3,
        help = "number of videos to skip after every video is processed",
        type = int,
    )
    parser.add_argument(
        "--frame_skip",
        default = 30,
        help = "number of frames to skip per video after every frame is processed",
        type = int,
    )
    parser.add_argument(
        "--confidence_threshold",
        default = 0.4,
        help = "confidence threshold of the model",
        type = float,
    )
    parser.add_argument(
        "--iou_threshold",
        default = "0.3",
        help = "iou threshold",
        type = float,
    )
    args = parser.parse_args()
    yolo_dataset = CreateYOLODataset(
        source_weight_path = args.source_weight_path,
        source_video_path = args.source_video_path,
        destination_path = args.destination_path,
        video_skip = args.video_skip,
        frame_skip = args.frame_skip,
        confidence_threshold = args.confidence_threshold,
        iou_threshold = args.iou_threshold
    )
    yolo_dataset.process()

"""
python core/yolo_model/create_yolo_dataset.py \
--source_weight_path checkpoints/rf-detr-nano.pth \
--source_video_path BRB_Dataset \
--destination_path BRB_YOLO_Dataset \
--video_skip 3 \
--frame_skip 40 \
--confidence_threshold 0.4 \
--iou_threshold 0.3
"""