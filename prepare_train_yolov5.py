import gdown
from pathlib import Path
import zipfile
import pandas as pd
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import ast
from tqdm.auto import tqdm
import PIL.Image as Image
import os
from utils.read import read_config

config = read_config()
cfg_yolo = config['detection']

data_url = cfg_yolo['data_url']
dataset_version =  cfg_yolo['dataset_version']

root_dir= 'data/{}'.format(dataset_version)
csv_file= 'data/{}/data_info.csv'.format(dataset_version)

gdown.download(data_url, "data.zip", quiet=False)

with zipfile.ZipFile("data.zip","r") as zip_ref:
        zip_ref.extractall('data')

df = pd.read_csv(csv_file)

## Convert to YOLO format

categories = list(df['label'].unique())
categories.sort()

data = df['file_name'].unique()
data = list(map(lambda x: {"name":x}, data))

for f in tqdm(data):
  lbls = df[df["file_name"]==f['name']]['label'].values
  flags = df[df["file_name"]==f['name']]['flag'].values
  bnd_bxs = df[df["file_name"]==f['name']]['tot_bnd_bxs'].values #[xmin, ymin, xmax, ymax]
  bnd_bxs = list(map(lambda x: ast.literal_eval(x), bnd_bxs))
  f['lbls'] = lbls
  f['flags'] = flags
  f['bnd_bxs'] = bnd_bxs

train_data, val_data = train_test_split(data, test_size=0.1) 

def create_dataset(data, root_dir, categories, dataset_type):
  images_path = Path(f"data/images/{dataset_type}")
  images_path.mkdir(parents=True, exist_ok=True)
  labels_path = Path(f"data/labels/{dataset_type}")
  labels_path.mkdir(parents=True, exist_ok=True)
  for img_id, row in enumerate(tqdm(data)):
    image_name = f"{img_id}.jpeg"
    img = row["name"]
    img = Image.open(os.path.join(root_dir, img))
    img = img.convert("RGB")
    width, height = img.size
    img.save(str(images_path / image_name), "JPEG")
    label_name = f"{img_id}.txt"
    with (labels_path / label_name).open(mode="w") as label_file:
      for i in range(len(row["lbls"])):
        category_idx = categories.index(row["lbls"][i])
        x1, y1, x2, y2 = row["bnd_bxs"][i]
        bbox_width = x2 - x1
        x_center = x1 + bbox_width / 2
        bbox_height = y2 - y1
        y_center = y1 + bbox_height / 2

        bbox_width /= width
        x_center /= width
        bbox_height /= height
        y_center /= height

        label_file.write(
          f"{category_idx} {x_center} {y_center} {bbox_width} {bbox_height}\n"
        )

create_dataset(train_data, root_dir, categories, "train")

create_dataset(val_data, root_dir, categories, "val")

