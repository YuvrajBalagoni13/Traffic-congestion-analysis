import os
from tqdm.auto import tqdm
import shutil

def Format_Dataset(
        dataset_dir: str = "BRB_YOLO_Dataset",
        train_val_test_split: int = 0.2
        ) -> None:
    image_dir = os.path.join(dataset_dir, "images")
    label_dir = os.path.join(dataset_dir, "labels")

    image_train_dir = os.path.join(image_dir, "train")
    image_val_dir = os.path.join(image_dir, "val")
    image_test_dir = os.path.join(image_dir, "test")
    os.makedirs(image_train_dir, exist_ok=True)
    os.makedirs(image_val_dir, exist_ok=True)
    os.makedirs(image_test_dir, exist_ok=True)

    label_train_dir = os.path.join(label_dir, "train")
    label_val_dir = os.path.join(label_dir, "val")
    label_test_dir = os.path.join(label_dir, "test")
    os.makedirs(label_train_dir, exist_ok=True)
    os.makedirs(label_val_dir, exist_ok=True)
    os.makedirs(label_test_dir, exist_ok=True)

    labels = os.listdir(label_dir)

    total_images = len(labels)
    train_len = int(total_images * (1 - 2 * train_val_test_split))
    val_len = int(train_len + int(total_images * train_val_test_split))
    test_len = int(val_len + int(total_images * train_val_test_split))
    print(f"{train_len} - {val_len} - {test_len}")

    count = 0
    for label in tqdm(labels):
        image_name = f"{label.split('.')[0]}.jpg"
        label_path = os.path.join(label_dir, label)
        if count <= train_len:
            shutil.move(os.path.join(image_dir, image_name), os.path.join(image_train_dir, image_name))
            shutil.move(os.path.join(label_dir, label), os.path.join(label_train_dir, label))
        elif train_len < count <= val_len:
            shutil.move(os.path.join(image_dir, image_name), os.path.join(image_val_dir, image_name))
            shutil.move(os.path.join(label_dir, label), os.path.join(label_val_dir, label))
        else:
            shutil.move(os.path.join(image_dir, image_name), os.path.join(image_test_dir, image_name))
            shutil.move(os.path.join(label_dir, label), os.path.join(label_test_dir, label))
        count += 1
    print(f"train dir len - {len(os.listdir(image_train_dir))}")
    print(f"val dir len - {len(os.listdir(image_val_dir))}")
    print(f"test dir len - {len(os.listdir(image_test_dir))}")
        
def remove_no_detection_samples(
        dataset_dir: str = "BRB_YOLO_Dataset",
        no_detection_dir: str = "No_detection_images"
        ):
    label_dir = os.path.join(dataset_dir, "labels")
    image_dir = os.path.join(dataset_dir, "images")
    destination_image_dir = os.path.join(no_detection_dir, "images")
    os.makedirs(no_detection_dir, exist_ok=True)
    os.makedirs(destination_image_dir, exist_ok=True)

    labels = os.listdir(label_dir)
    no_detections = 0
    detection = 0
    for label in tqdm(labels):
        label_path = os.path.join(label_dir, label)
        image_name = f"{label.split('.')[0]}.jpg"
        if os.path.getsize(label_path) == 0:
            save_name = f"{no_detections:06d}.jpg"
            os.remove(os.path.join(label_path))
            shutil.move(os.path.join(image_dir, image_name), os.path.join(destination_image_dir, save_name))
            no_detections += 1
            continue
        save_name = f"{detection:06d}.jpg"
        save_label_name = f"{detection:06d}.txt"
        os.rename(os.path.join(image_dir, image_name), os.path.join(image_dir, save_name))
        os.rename(label_path, os.path.join(label_dir, save_label_name))
        detection += 1
    print(no_detections)

Format_Dataset()