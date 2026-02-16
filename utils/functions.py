import os
from tqdm.auto import tqdm

dataset_label_path = "BRB_YOLO_Dataset/labels"

def check_without_detections(label_dir):
    labels = os.listdir(label_dir)
    no_detections = 0

    for label in tqdm(labels):
        label_path = os.path.join(label_dir, label)

        if os.path.getsize(label_path) == 0:
            no_detections += 1

    with_detections = len(labels) - no_detections
    print(f"Total: {len(labels)} | Empty: {no_detections} | Valid: {with_detections}")

def convert_all_classes_to_one(label_dir):
    for label in tqdm(os.listdir(label_dir)):
        label_path = os.path.join(label_dir, label)
        with open(label_path, "r") as f:
            lines = f.readlines()
        with open(label_path, "w") as f:
            for line in lines:
                parts = line.split()
                parts[0] = "0"
                f.write(" ".join(parts) + "\n")

check_without_detections(dataset_label_path)