import os
import json
import torch
import numpy as np
from PIL import Image
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.dataset import Dataset


def parse_annotations(annotation_file):
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)
    return annotations


def get_category_id(annotations, category_name):
    for category in annotations['categories']:
        if category['name'] == category_name:
            return category['id']
    return None


def get_image_ids_with_category(annotations, category_ids):
    image_ids = set()
    for category_id in category_ids:
        for anno in annotations['annotations']:
            if anno['category_id'] == category_id:
                image_ids.add(anno['image_id'])
    return image_ids


def get_id_filename_mapping(annotations):
    mapping = {}
    for image in annotations['images']:
        mapping[image['id']] = image['file_name']
    return mapping


def get_filenames(image_ids, id_filename_mapping):
    filenames = []
    for image_id in image_ids:
        filenames.append(id_filename_mapping[image_id])
    return filenames


def get_coco_data(normal_class_id):
    root_dir = "../Data/Atika/Model_Optimization/data/"
    # train_dir = os.path.join(root_dir, 'train2017')
    # val_dir = os.path.join(root_dir, 'val2017')

    # Parse annotations
    train_annotations = parse_annotations(os.path.join(root_dir, 'annotations/instances_train2017.json'))
    val_annotations = parse_annotations(os.path.join(root_dir, 'annotations/instances_val2017.json'))

    # Get category ID for normal_class
    # normal_class_id = get_category_id(train_annotations, normal_class)

    # Get image IDs with the normal class
    normal_image_ids = get_image_ids_with_category(train_annotations, normal_class_id)

    # Create an ID-to-filename mapping
    train_id_filename_mapping = get_id_filename_mapping(train_annotations)

    # Get normal filenames
    normal_filenames = get_filenames(normal_image_ids, train_id_filename_mapping)

    # Get all train filenames
    all_train_filenames = [img['file_name'] for img in train_annotations['images']]

    # Outlier filenames are those not in normal_filenames
    outlier_filenames = list(set(all_train_filenames) - set(normal_filenames))

    return normal_filenames, outlier_filenames


class COCO_Dataset(Dataset):
    def __init__(self, traindir, imagenames, labels):
        self.traindir = traindir
        self.imagenames = imagenames
        self.labels = labels
        self.transformations = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
            
    def _load_image(self, idx):
        img_path = os.path.join(self.traindir, self.imagenames[idx])
        img = Image.open(img_path).convert("RGB")  # Ensuring 3 channels
        if self.transformations:
            img = self.transformations(img)
        return img, self.labels[idx]

    def __getitem__(self, idx):
        return self._load_image(idx)

    def __len__(self):
        return len(self.imagenames)
