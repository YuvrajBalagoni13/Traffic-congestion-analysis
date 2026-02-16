import requests
import os
import pandas as pd
from tqdm.auto import tqdm

data_len = 300
data_path = "BRB_Dataset"
os.makedirs(data_path, exist_ok=True)

parent_folders = ["normanniles1", "normanniles2", "normanniles3", "normanniles4"]
bucket_list_url = "https://storage.googleapis.com/brb-traffic"
train_data = pd.read_csv("zipfile\Train.csv")

filename_data = []
for filename in train_data["videos"][data_len:(1160)]:
    filename_data.append(filename.split("/")[-1])

for parent_folder in parent_folders:
    parent_folder_path = os.path.join(data_path, parent_folder)
    os.makedirs(parent_folder_path, exist_ok=True)
    
    print(f"downloading for {parent_folder_path} ...")
    print("-" * 50)
    
    for file in tqdm(filename_data):

        if parent_folder != file.split("_")[0]:
            file_name_parts = file.split("_")
            file_name_parts[0] = parent_folder
            file = "_".join(file_name_parts)

        file_url = f"{bucket_list_url}/{parent_folder}/{file}"
        local_filename = os.path.join(parent_folder_path, file)

        if os.path.exists(local_filename) ==False:
            try:
                with requests.get(file_url, stream=True) as r:
                    r.raise_for_status()
                    with open(local_filename, "wb") as f:
                        for chunk in r.iter_content(chunk_size=10485760):
                            f.write(chunk)
            except:
                print(f"{file_url} does not exist")
        else:
            continue
    
    print(f"Done downloading for {parent_folder_path}")

print("Completed Download")