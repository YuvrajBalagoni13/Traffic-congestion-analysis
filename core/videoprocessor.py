import argparse
import json
import numpy as np
import pandas as pd
import supervision as sv
from rfdetr import RFDETRNano, RFDETRSmall, RFDETRMedium, RFDETRLarge
from rfdetr.util.coco_classes import COCO_CLASSES
from ultralytics import YOLO
import cv2
from time import time
from tqdm.auto import tqdm

class VideoProcessor:
    def __init__(
            self,
            source_weights_path: str,
            confidence_threshold: float = 0.3,
            iou_threshold: float = 0.7
    ) -> None:
        self.source_weights_path = source_weights_path
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold

        # self.model = RFDETRNano()
        # try:
        #     self.model.optimize_for_inference(compile=True, batch_size=1)
        #     print("Model optimized successfully")
        # except Exception as e:
        #     print(f"Optimization failed, falling back to standard inference. Error: {e}")

        self.model_type = "rfdetr" if source_weights_path.split("/")[-1].split("-")[0] == "rf" else "yolo"

        if self.model_type == "yolo":
            self.model = YOLO(self.source_weights_path)
        else:
            self.model = RFDETRMedium(pretrain_weights=self.source_weights_path)
            try:
                self.model.optimize_for_inference(compile=False, batch_size=1)
                print("Model optimized successfully")
            except Exception as e:
                print(f"Optimization failed, falling back to standard inference. Error: {e}")

        self.tracker = sv.ByteTrack()
    
        self.polygon_coordinates = {
            "normanniles1" : {
                "Entry": np.array([[424, 199], [590, 205], [885, 291], [848, 365], [425, 308]]),
                "Exit" : np.array([[425, 255], [129, 258], [316, 198], [424, 199]]),
                "Roundabout_intersection": np.array([sv.Point(x=425, y=199), sv.Point(x=425, y=400)])
            },
            "normanniles2" : {
                "Entry" : np.array([[376, 200], [558, 237], [522, 323], [258, 322], [258, 204]]),
                "Exit" : np.array([[258, 256], [1, 265], [151, 206], [258, 204]]),
                "Roundabout_intersection": np.array([sv.Point(x=258, y=204), sv.Point(x=258, y=400)])
            },
            "normanniles3" : {
                "Entry" : np.array([[376, 162], [567, 172], [765, 236], [694, 312], [376, 290]]),
                "Exit" : np.array([[376, 162], [253, 161], [68, 223], [376, 224]]),
                "Roundabout_intersection": np.array([sv.Point(x=376, y=162), sv.Point(x=376, y=400)])
            },
            "normanniles4" : {
                "Entry" : np.array([[745, 212], [856, 266], [821, 342], [560, 333], [560, 254], [659, 204]]),
                "Exit" : np.array([[541, 261], [369, 236], [589, 196], [658, 203]]),
                "Roundabout_intersection": np.array([sv.Point(x=560, y=250), sv.Point(x=560, y=400)]),
            }
        }

    def process_video(
            self,
            source_video_path: str,
            target_video_path: str = None,
            frame_skip: int = 0,
            track: bool = True
        ) -> dict:
        self.source_video_path = source_video_path
        self.target_video_path = target_video_path
        self.frame_skip = frame_skip
        self.camera = self.source_video_path.split("/")[-1].split("_")[0]

        self.feature_output = {}

        self.video = cv2.VideoCapture(source_video_path)

        if not self.video.isOpened():
            print("Video not found")
            return None

        self.frame_width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.video.get(cv2.CAP_PROP_FPS) if self.video.get(cv2.CAP_PROP_FPS) > 0 else 20.0
        self.total_frames = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))

        self.entry_polygon_zone = sv.PolygonZone(
            polygon = self.polygon_coordinates[self.camera]["Entry"]
        )
        self.exit_polygon_zone = sv.PolygonZone(
            polygon = self.polygon_coordinates[self.camera]["Exit"]
        )
        self.roundabout_intersection_line = sv.LineZone(
            start = self.polygon_coordinates[self.camera]["Roundabout_intersection"][0],
            end = self.polygon_coordinates[self.camera]["Roundabout_intersection"][1]
        )
        pbar = tqdm(total=self.total_frames, desc="Processing Video")

        if self.target_video_path is not None:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(self.target_video_path, fourcc, self.fps, (self.frame_width, self.frame_height))

        entry_count, exit_count = 0, 0
        frame_count = 1

        start_time = time()
        while self.video.isOpened():
            ret, frame = self.video.read()
            if not ret or frame is None:
                print("End of video stream or failed to read frame.")
                break

            if frame_count % (frame_skip + 1) == 0:
                image = frame.copy()
                detections = self.process_frame(image, track=track)

                if track:
                    self.extract_features(frame_count, detections)

                if self.target_video_path is not None:
                    annotated_frame = self.annotate_frame(image, detections, track)
                    out.write(annotated_frame)

            pbar.update(1)
            frame_count += 1

        end_time = time()

        pbar.close()
        self.video.release()
        if self.target_video_path is not None:
            out.release()
        cv2.destroyAllWindows()  

        return self.feature_output 

    def extract_features(
            self,
            frame_count: int,
            detections: sv.Detections
    ) -> None:
       
        for i in range(len(detections)):

            tracker_id = detections.tracker_id[i].item()
            detection = detections[i]

            if tracker_id in self.feature_output:
                dict = self.feature_output[tracker_id]
            else:
                dict = {}
                dict["zone"] = None
                dict["center_position"] = [10000, 10000]
                dict["idle_frames"] = 0
                dict["in_roundabout"] = False
                dict["first_detect"] = 10000
                dict["last_detect"] = 0
                dict["entry_frame"] = 10000
                dict["exit_frame"] = 0

            if frame_count <= dict["first_detect"]:
                dict["first_detect"] = frame_count
            if frame_count > dict["last_detect"]:
                dict["last_detect"] = frame_count

            bbox = detection.xyxy[0]
            center_x, center_y = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2

            in_enter = self.entry_polygon_zone.trigger(detection)
            in_exit = self.exit_polygon_zone.trigger(detection)
            crossed_in, crossed_out = self.roundabout_intersection_line.trigger(detection)

            if abs(center_x - dict["center_position"][0]) + abs(center_y - dict["center_position"][1]) <= (4 * (self.frame_skip + 1)) and (in_enter or in_exit):
                dict["idle_frames"] += 1
            dict["center_position"] = [center_x.item(), center_y.item()]

            if crossed_in:
                dict["in_roundabout"] = True

            if in_enter and dict["zone"] == None:
                dict["zone"] = "entry"
            elif in_exit and dict["zone"] == None:
                dict["zone"] = "exit"

            if frame_count <= dict["entry_frame"] and (in_enter or in_exit):
                dict["entry_frame"] = frame_count
            if frame_count >= dict["exit_frame"] and (in_enter or in_exit):
                dict["exit_frame"] = frame_count

            self.feature_output[tracker_id] = dict

    def create_feature_row(
            self
    ) -> dict:
        """
        All the features ->
                1. video_name : str 
                2. Camera : str 
                3. Time : float
                4. No. of vehicles in entry : int
                5. No.of vehicles in roundabout : int
                6. Avg time spent in entry by vehicles coming from roundabout : float
                7. Average Idle time entry : float
                8. Average time spent in entry : float
                9. idle vehicles 0-150 : int
                10. idle vehicles 150-300 : int
                11. idle vehicles 300-450 : int
                12. idle vehicles 450-600 : int
                13. idle vehicles 600-inf : int
                14. time spent in entry 0-150 : int
                15. time spent in entry 150-300 : int
                16. time spent in entry 300-450 : int
                17. time spent in entry 450-600 : int
                18. time spent in entry 600-inf : int
        """
        video_name = self.source_video_path.split("/")[-1]
        feature_dict = {}
        feature_dict['video_name'] = video_name
        feature_dict['camera'] = video_name.split("_")[0]

        date_time_arr = video_name.split("_")[-1].split(".")[0].split("-")
        date = "-".join(date_time_arr[:3])
        time = ":".join(date_time_arr[3:])
        feature_dict['date_time'] = f"{date} {time}"

        entry_count, exit_count = 0, 0
        roundabout_vehicle_count = 0
        avg_time_spent_entry, avg_time_spent_exit = 0, 0
        avg_time_spent_from_roundabout = 0
        avg_idle_time_entry, avg_idle_time_exit = 0, 0
        avg_idle_time_from_roundabout = 0
        idle_arr_entry = [0, 0, 0, 0, 0]        # <150, 150-300, 300-450, 450-600, 600<
        idle_arr_exit = [0, 0, 0, 0, 0]         # <150, 150-300, 300-450, 450-600, 600<
        time_spent_arr_entry = [0, 0, 0, 0, 0]  # <150, 150-300, 300-450, 450-600, 600<
        time_spent_arr_exit = [0, 0, 0, 0, 0]   # <150, 150-300, 300-450, 450-600, 600<

        for key, val in self.feature_output.items():
            time_spent = val["exit_frame"] - val["entry_frame"]
            if abs(val["first_detect"] - val["last_detect"]) <= 30:
                continue
            idle_time = val["idle_frames"]

            if val["zone"] == "entry":
                entry_count += 1
                avg_time_spent_entry += time_spent

                if val["in_roundabout"]:
                    roundabout_vehicle_count += 1
                    avg_time_spent_from_roundabout += time_spent
                    avg_idle_time_from_roundabout += idle_time
                
                avg_idle_time_entry += idle_time
                idle_arr_entry[int(min(idle_time // 50, 4))] += 1
                time_spent_arr_entry[int(min(time_spent // 50, 4))] += 1                                    

            if val["zone"] == "exit":
                exit_count += 1
                avg_time_spent_exit += time_spent
                avg_idle_time_exit += idle_time
                idle_arr_exit[int(min(idle_time // 50, 4))] += 1
                time_spent_arr_exit[int(min(time_spent // 50, 4))] += 1
            
        feature_dict['number_of_vehicles_entry'] = entry_count
        feature_dict['number_of_vehicles_exit'] = exit_count
        feature_dict['number_of vehicles_roundabout'] = roundabout_vehicle_count

        feature_dict['avg_time_spent_entry'] = avg_time_spent_entry / len(self.feature_output) if len(self.feature_output) != 0 else 0
        feature_dict['avg_time_spent_exit'] = avg_time_spent_exit / len(self.feature_output) if len(self.feature_output) != 0 else 0
        feature_dict['avg_time_spent_from_roundabout'] = avg_time_spent_from_roundabout / len(self.feature_output) if len(self.feature_output) != 0 else 0

        feature_dict['avg_idle_time_entry'] = avg_idle_time_entry / len(self.feature_output) if len(self.feature_output) != 0 else 0
        feature_dict['avg_idle_time_exit'] = avg_idle_time_exit / len(self.feature_output) if len(self.feature_output) != 0 else 0
        feature_dict['avg_idle_time_from_roundabout'] = avg_idle_time_from_roundabout / len(self.feature_output) if len(self.feature_output) != 0 else 0

        for idx, val in enumerate(idle_arr_entry):
            feature_dict[f'idle_{idx}_entry'] = val
        for idx, val in enumerate(idle_arr_exit):
            feature_dict[f'idle_{idx}_exit'] = val

        for idx, val in enumerate(time_spent_arr_entry):
            feature_dict[f'time_spent_{idx}_entry'] = val
        for idx, val in enumerate(time_spent_arr_exit):
            feature_dict[f'time_spent_{idx}_exit'] = val

        return feature_dict


    def annotate_frame(
            self,
            frame: np.ndarray,
            detections: sv.Detections,
            track: bool = True
    ) -> np.ndarray:
        
        if detections.xyxy.size == 0:
            return frame
        
        if track:
            labels = [
                f"#{tracker_id}"
                for tracker_id in detections.tracker_id
            ]
        else:
            labels = [
                f"{COCO_CLASSES[class_id]} {confidence:.2f}"
                for class_id, confidence
                in zip(detections.class_id, detections.confidence)
            ]

        annotated_image = sv.BoxAnnotator().annotate(frame, detections)
        annotated_image = sv.LabelAnnotator().annotate(annotated_image, detections, labels)
        annotated_image = sv.PolygonZoneAnnotator(self.entry_polygon_zone, sv.Color.GREEN).annotate(annotated_image)
        annotated_image = sv.PolygonZoneAnnotator(self.exit_polygon_zone, sv.Color.RED).annotate(annotated_image)
        annotated_image = sv.LineZoneAnnotator(thickness=2, text_thickness=1, text_scale=0.5).annotate(annotated_image, self.roundabout_intersection_line)
        return annotated_image

    def process_frame(
            self,
            frame: np.ndarray,
            track: bool = True
    ) -> np.ndarray:
        if self.model_type == "yolo":
            results = self.model.predict(frame, conf=self.confidence_threshold, verbose=False)[0]
            detections = sv.Detections.from_ultralytics(results)
        else:
            detections = self.model.predict(frame, threshold=self.confidence_threshold)
        if track:
            detections = self.tracker.update_with_detections(detections)
        return detections

if __name__=="__main__":
    parser = argparse.ArgumentParser(
        description="BRB - Traffic roundabout analysis"
    )

    parser.add_argument(
        "--source_weights_path",
        required=True,
        help="Path to source weight file",
        type=str,
    )

    parser.add_argument(
        "--source_video_path",
        required=True,
        help="Path to source video file",
        type=str,
    )

    parser.add_argument(
        "--target_video_path",
        default=None,
        help="Path to target video file",
        type=str,
    )

    parser.add_argument(
        "--frame_skip",
        default=0,
        help="frames to skip after every frame processed",
        type=int,
    )

    parser.add_argument(
        "--confidence_threshold",
        default=0.3,
        help="Path to source weight file",
        type=float,
    )

    parser.add_argument(
        "--iou_threshold",
        default=0.7,
        help="Path to source weight file",
        type=float,
    )

    parser.add_argument(
        "--track",
        default=True,
        help="track or not",
        type=bool,
    )

    args = parser.parse_args()
    processor = VideoProcessor(
        source_weights_path=args.source_weights_path,
        confidence_threshold=args.confidence_threshold,
        iou_threshold=args.iou_threshold
    )
    feature_output = processor.process_video(
        source_video_path=args.source_video_path,
        target_video_path=args.target_video_path,
        frame_skip = args.frame_skip,
        track = args.track
    )
    print(feature_output)

    with open(f"{args.source_video_path.split('/')[-1].split('.')[0]}.json", "w") as f:
        json.dump(feature_output, f, indent=4)

"""
python core/videoprocessor.py \
--source_weights_path checkpoints/rf-detr-medium.pth \
--source_video_path BRB_Dataset/normanniles1/normanniles1_2025-10-20-11-36-45.mp4 \
--target_video_path output_feature_normanniles2_2025-10-20-11-36-45.mp4 \
--frame_skip 40 \
--confidence_threshold 0.4 \
--iou_threshold 0.3 \
--track True
"""