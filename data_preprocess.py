import os
import shutil
import random
import xml.etree.ElementTree as ET
import pandas as pd
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# Configuration
SEED = 42
SPLIT_RATIO = {'train': 0.7, 'val': 0.2, 'test': 0.1}
MAX_SAMPLES_PER_CLASS = 3000
OUTPUT_DIR = Path("datasets/AerialGuard")
CLASS_NAMES = ["UAV", "Aircraft", "Bird"]

# Set random seed for reproducibility
random.seed(SEED)


def create_directory_structure():
    """Creates the standard YOLOv8 directory tree."""
    for split in ['train', 'val', 'test']:
        (OUTPUT_DIR / 'images' / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / 'labels' / split).mkdir(parents=True, exist_ok=True)


def get_image_size(img_path):
    """Returns (width, height) of an image."""
    with Image.open(img_path) as img:
        return img.size


def convert_to_yolo(size, box):
    """Normalizes bounding box to YOLO format (x_center, y_center, width, height)."""
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[2]) / 2.0
    y = (box[1] + box[3]) / 2.0
    w = box[2] - box[0]
    h = box[3] - box[1]
    return x * dw, y * dh, w * dw, h * dh


def process_det_fly(root_path, class_id=0):
    """Processes Pascal VOC format for UAVs."""
    data = []
    xml_dir = Path(root_path) / "Annotations"
    img_dir = Path(root_path) / "Images"

    for xml_file in xml_dir.glob("*.xml"):
        img_path = img_dir / f"{xml_file.stem}.jpg"
        if not img_path.exists(): continue

        tree = ET.parse(xml_file)
        root = tree.getroot()
        size = (int(root.find('size/width').text), int(root.find('size/height').text))

        bboxes = []
        for obj in root.findall('object'):
            xmlbox = obj.find('bndbox')
            b = (float(xmlbox.find('xmin').text), float(xmlbox.find('ymin').text),
                 float(xmlbox.find('xmax').text), float(xmlbox.find('ymax').text))
            yolo_box = convert_to_yolo(size, b)
            bboxes.append(f"{class_id} " + " ".join([f"{a:.6f}" for a in yolo_box]))

        if bboxes: data.append((img_path, bboxes))
    return data


def process_fgvc_aircraft(root_path, csv_path, class_id=1):
    """Processes Custom CSV format for Aircrafts."""
    data = []
    img_dir = Path(root_path) / "images"
    df = pd.read_csv(csv_path)  # Expected: filename, xmin, ymin, xmax, ymax

    grouped = df.groupby('filename')
    for img_name, group in grouped:
        img_path = img_dir / img_name
        if not img_path.exists(): continue

        size = get_image_size(img_path)
        bboxes = []
        for _, row in group.iterrows():
            b = (row['xmin'], row['ymin'], row['xmax'], row['ymax'])
            yolo_box = convert_to_yolo(size, b)
            bboxes.append(f"{class_id} " + " ".join([f"{a:.6f}" for a in yolo_box]))

        data.append((img_path, bboxes))
    return data


def process_cub_200(root_path, class_id=2):
    """Processes CUB-200 TXT format for Birds."""
    data = []
    img_dir = Path(root_path) / "images"
    # CUB format: <image_id> <x> <y> <width> <height>
    bbox_file = Path(root_path) / "bounding_boxes.txt"
    images_file = Path(root_path) / "images.txt"

    img_map = {line.split()[0]: line.split()[1] for line in open(images_file)}

    with open(bbox_file, 'r') as f:
        for line in f:
            img_id, x, y, w, h = map(float, line.split())
            img_path = img_dir / img_map[str(int(img_id))]
            if not img_path.exists(): continue

            size = get_image_size(img_path)
            # CUB is already x,y,w,h (top-left). Convert to YOLO.
            dw, dh = 1. / size[0], 1. / size[1]
            yolo_box = ((x + w / 2) * dw, (y + h / 2) * dh, w * dw, h * dh)
            bboxes = [f"{class_id} " + " ".join([f"{a:.6f}" for a in yolo_box])]

            data.append((img_path, bboxes))
    return data


def save_dataset(dataset_list, split_name):
    """Saves images and labels to the designated YOLO split directory."""
    for img_path, bboxes in tqdm(dataset_list, desc=f"Saving {split_name}"):
        dest_img = OUTPUT_DIR / 'images' / split_name / img_path.name
        dest_lbl = OUTPUT_DIR / 'labels' / split_name / f"{img_path.stem}.txt"

        shutil.copy(img_path, dest_img)
        with open(dest_lbl, 'w') as f:
            f.write("\n".join(bboxes))


def main():
    create_directory_structure()

    # 1. Collect and Downsample
    print("Collecting data from sources...")
    uav_data = process_det_fly("path/to/det_fly")
    air_data = process_fgvc_aircraft("path/to/fgvc", "path/to/fgvc/anno.csv")
    bird_data = process_cub_200("path/to/cub200")

    all_data = []
    for dataset in [uav_data, air_data, bird_data]:
        random.shuffle(dataset)
        all_data.append(dataset[:MAX_SAMPLES_PER_CLASS])

    # 2. Split into Train, Val, Test per class to maintain balance in splits
    final_splits = {'train': [], 'val': [], 'test': []}

    for class_subset in all_data:
        n = len(class_subset)
        train_idx = int(n * SPLIT_RATIO['train'])
        val_idx = train_idx + int(n * SPLIT_RATIO['val'])

        final_splits['train'].extend(class_subset[:train_idx])
        final_splits['val'].extend(class_subset[train_idx:val_idx])
        final_splits['test'].extend(class_subset[val_idx:])

    # 3. Save files
    for split in ['train', 'val', 'test']:
        random.shuffle(final_splits[split])
        save_dataset(final_splits[split], split)

    # 4. Generate data.yaml
    yaml_content = f"""
path: {OUTPUT_DIR.absolute()}
train: images/train
val: images/val
test: images/test

names:
  0: UAV
  1: Aircraft
  2: Bird
"""
    with open(OUTPUT_DIR / "data.yaml", "w") as f:
        f.write(yaml_content)

    print(f"Dataset preparation complete. Path: {OUTPUT_DIR}")


if __name__ == "__main__":
    # Ensure you update paths to actual source directories before running
    main()
