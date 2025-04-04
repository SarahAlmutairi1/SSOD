# Pascal VOC Download script
import xml.etree.ElementTree as ET
from tqdm import tqdm
from ultralytics.utils.downloads import download
import os
from pathlib import Path
from shutil import move
from config import HOME
import preprocess

def convert_label(path, lb_path, year, image_id):
    def convert_box(size, box):
        dw, dh = 1. / size[0], 1. / size[1]
        x, y, w, h = (box[0] + box[1]) / 2.0 - 1, (box[2] + box[3]) / 2.0 - 1, box[1] - box[0], box[3] - box[2]
        return x * dw, y * dh, w * dw, h * dh

    in_file = open(path / f'VOC{year}/Annotations/{image_id}.xml')
    out_file = open(lb_path, 'w')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    names = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
    ]
    for obj in root.iter('object'):
        cls = obj.find('name').text
        if cls in names and int(obj.find('difficult').text) != 1:
            xmlbox = obj.find('bndbox')
            bb = convert_box((w, h), [float(xmlbox.find(x).text) for x in ('xmin', 'xmax', 'ymin', 'ymax')])
            cls_id = names.index(cls)  # class id
            out_file.write(" ".join(str(a) for a in (cls_id, *bb)) + '\n')


def download_VOC():
    # Download path
    dir = Path(HOME)/ "src" / "datasets" / "VOC"
    url = 'https://github.com/ultralytics/assets/releases/download/v0.0.0/'
    urls = [f'{url}VOCtrainval_06-Nov-2007.zip',  # 446MB, 5012 images
            f'{url}VOCtest_06-Nov-2007.zip',  # 438MB, 4953 images
            f'{url}VOCtrainval_11-May-2012.zip']  # 1.95GB, 17126 images
    download(urls, dir=dir / 'images', curl=True, threads=3, exist_ok=True)  # download and unzip over existing paths (required)
    
    # Convert
    path = dir / 'images/VOCdevkit'
    for year, image_set in ('2012', 'train'), ('2012', 'val'), ('2007', 'train'), ('2007', 'val'), ('2007', 'test'):
        imgs_path = dir / 'images' / f'{image_set}{year}'
        lbs_path = dir / 'labels' / f'{image_set}{year}'
        imgs_path.mkdir(exist_ok=True, parents=True)
        lbs_path.mkdir(exist_ok=True, parents=True)
    
        with open(path / f'VOC{year}/ImageSets/Main/{image_set}.txt') as f:
            image_ids = f.read().strip().split()
        for id in tqdm(image_ids, desc=f'{image_set}{year}'):
            f = path / f'VOC{year}/JPEGImages/{id}.jpg'  # old img path
            lb_path = (lbs_path / f.name).with_suffix('.txt')  # new label path
            f.rename(imgs_path / f.name)  # move image
            convert_label(path, lb_path, year, id)  # convert labels to YOLO format
    
    
    # Paths
    base_dir = Path(HOME)/ "src" / "datasets" / "VOC"
    current_dir_img = base_dir / "images"
    current_dir_label = base_dir / "labels"
    output_dir = Path(HOME)/ "src" / "datasets" / "VOC1"
    
    # Create new structure
    for split in ["train", "valid", "test"]:
        (output_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / split / "labels").mkdir(parents=True, exist_ok=True)
    
    # Mapping for splits
    splits_mapping = {
        "train2007": "train",
        "train2012": "valid",
        "val2007": "train",
        "val2012": "valid",
        "test2007": "test",
    }
    
    # Move files
    for split_folder, split in splits_mapping.items():
        image_folder = current_dir_img / split_folder
        label_folder = current_dir_label / split_folder
    
        # Check if paths exist
        if not image_folder.exists() or not label_folder.exists():
            print(f"Skipping {image_folder}")
            print(f"Skipping {label_folder}")
            print(f"Skipping {split_folder}: Missing images or labels.")
            continue
    
        # Move images
        for img_file in tqdm(image_folder.glob("*.jpg"), desc=f"Moving images for {split_folder}"):
            move(str(img_file), str(output_dir / split / "images" / img_file.name))
    
        # Move labels
        for anno_file in tqdm(label_folder.glob("*.txt"), desc=f"Moving labels for {split_folder}"):
            new_label_path = output_dir / split / "labels" / anno_file.name
            move(str(anno_file), str(new_label_path))
    
    # Create data.yaml
    yaml_content = f"""
    path: VOC1
    train: train/images
    val: valid/images
    test: test/images
    
    names:
      0: aeroplane
      1: bicycle
      2: bird
      3: boat
      4: bottle
      5: bus
      6: car
      7: cat
      8: chair
      9: cow
      10: diningtable
      11: dog
      12: horse
      13: motorbike
      14: person
      15: pottedplant
      16: sheep
      17: sofa
      18: train
      19: tvmonitor
    """
    
    with open(output_dir / "data.yaml", "w") as f:
        f.write(yaml_content)
    
    print("Restructuring complete")
