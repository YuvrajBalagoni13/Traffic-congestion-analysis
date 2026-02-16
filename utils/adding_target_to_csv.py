import pandas as pd

train_dataset = pd.read_csv("zipfile\Train.csv")
dataset_csv = pd.read_csv("dataset_features_yolo26.csv", index_col=0)

train_dataset["video_name"] = train_dataset["videos"].apply(lambda x: x.split("/")[-1])

final_dataset = pd.merge(
    dataset_csv,
    train_dataset[["video_name", "signaling", "congestion_enter_rating", "congestion_exit_rating"]],
    on="video_name",
    how="left"
)

final_dataset.to_csv("final_updated_dataset_features.csv", index=False)