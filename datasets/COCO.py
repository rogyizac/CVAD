import os
import json
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.dataset import Dataset

import albumentations as A
from albumentations.pytorch import ToTensorV2

from utils.bert_utils import init_bert_model

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
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        self.albumentations_transform = A.Compose([
                                                    A.Resize(height=256, width=256),
                                                    A.HorizontalFlip(p=0.5),
                                                    # A.VerticalFlip(p=0.5), 
                                                    A.Rotate(limit=30, p=0.5),
                                                    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                                                    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.5),
                                                ])
        
            
    def _load_image(self, idx):
        img_path = os.path.join(self.traindir, self.imagenames[idx])
        img = Image.open(img_path).convert("RGB")  # Ensuring 3 channels
        img = np.array(img)
        if self.albumentations_transform:
            img = self.albumentations_transform(image=img)["image"]
        
        to_tensor_transform = transforms.ToTensor()
        img = to_tensor_transform(img)
        
        if self.transformations:
            img = self.transformations(img)
        
        return img, self.labels[idx]

    def __getitem__(self, idx):
        return self._load_image(idx)

    def __len__(self):
        return len(self.imagenames)

    
def get_coco_image_captions_data(normal_class):
    
    """
    Create a dataframe for image and text input in the format
    
    target images, related captions
    other images, related captions
    target images, unrelated captions
    other images, unrelated captions
    """
    
    if os.path.exists("coco_image_captions.csv"):
        df = pd.read_csv(r"coco_image_captions.csv")    
    else:
        root_dir = "../Data/Atika/Model_Optimization/data/"
        df_train = create_coco_dataframe(os.path.join(root_dir, 'annotations/instances_train2017.json'), os.path.join(root_dir, 'annotations/captions_train2017.json'))
        df_val = create_coco_dataframe(os.path.join(root_dir, 'annotations/instances_val2017.json'), os.path.join(root_dir, 'annotations/captions_val2017.json'))
        df_train['folder'] = 'train2017'
        df_val['folder'] = 'val2017'

        ### target images, related captions | other images, related captions
        df = pd.concat([df_train, df_val])
        df['caption_label'] = 0
        df['image_label'] = df['labels'].apply(lambda labels: 0 if normal_class in labels else 1)

        ### target images, unrelated captions
        # Filter for images label 1 present
        df_label_present = df[df['image_label'] == 0].reset_index(drop=True)

        # Precompute all captions and create a mapping to exclude the original caption more efficiently
        all_captions = df['caption'].tolist()

        # Generate a random index list for captions, ensuring not to match with its own index
        random_indices = np.random.randint(0, len(df), size=len(df_label_present))

        # Initialize lists for the new DataFrame
        new_image_ids = df_label_present['image_id'].tolist()
        new_image_paths = df_label_present['image_path'].tolist()
        new_image_labels = df_label_present['image_label'].tolist()
        new_labels = df_label_present['labels'].tolist()
        new_folder = df_label_present['folder'].tolist()

        # Efficiently select random captions, excluding the original by checking and re-selecting if necessary
        new_captions = [all_captions[idx] if idx != row_index else all_captions[(idx + 1) % len(all_captions)]
                        for row_index, idx in enumerate(random_indices)]

        # Create the new DataFrame
        new_df_with_mismatched_captions_label_present = pd.DataFrame({
            'image_id': new_image_ids,
            'image_path': new_image_paths,
            'caption': new_captions,
            'labels': new_labels,
            'image_label': new_image_labels,
            'caption_label': [1] * len(new_image_ids),
            'folder': new_folder
        })

        ### other images, unrelated captions
        df_label_absent = df[df['image_label'] == 1].reset_index(drop=True)

        # Precompute all captions and create a mapping to exclude the original caption more efficiently
        all_captions = df['caption'].tolist()

        # Generate a random index list for captions, ensuring not to match with its own index
        random_indices = np.random.randint(0, len(df), size=len(df_label_absent))

        # Initialize lists for the new DataFrame
        new_image_ids = df_label_absent['image_id'].tolist()
        new_image_paths = df_label_absent['image_path'].tolist()
        new_image_labels = df_label_absent['image_label'].tolist()
        new_labels = df_label_absent['labels'].tolist()
        new_folder = df_label_absent['folder'].tolist()

        # Efficiently select random captions, excluding the original by checking and re-selecting if necessary
        new_captions = [all_captions[idx] if idx != row_index else all_captions[(idx + 1) % len(all_captions)]
                        for row_index, idx in enumerate(random_indices)]

        # Create the new DataFrame
        new_df_with_mismatched_captions_label_absent = pd.DataFrame({
            'image_id': new_image_ids,
            'image_path': new_image_paths,
            'caption': new_captions,
            'labels': new_labels,
            'image_label': new_image_labels,
            'caption_label': [1] * len(new_image_ids),
            'folder': new_folder
        })

        df = pd.concat([df, new_df_with_mismatched_captions_label_absent, new_df_with_mismatched_captions_label_present])
        df = df.reset_index(drop = True)
        df = df.sample(frac = 1)
        df.to_csv(r"coco_image_captions.csv", index = False)
        
    return df
    
def create_coco_dataframe(image_annotations, caption_annotations):
    # Parse the annotations
    images_data = parse_annotations(image_annotations)
    captions_data = parse_annotations(caption_annotations)

    # Create a mapping from image_id to file name for easier access
    id_filename_mapping = get_id_filename_mapping(images_data)

    # Create a mapping from image_id to labels to avoid nested loops
    image_id_to_labels = {}
    for anno in images_data['annotations']:
        image_id = anno['image_id']
        if image_id not in image_id_to_labels:
            image_id_to_labels[image_id] = []
        image_id_to_labels[image_id].append(anno['category_id'])

    # Create empty list to store dataframe rows
    rows = []

    # Iterate through captions and match with images
    for caption in captions_data['annotations']:
        image_id = caption['image_id']
        if image_id in id_filename_mapping:
            image_path = id_filename_mapping[image_id]
            caption_text = caption['caption']
            labels = image_id_to_labels.get(image_id, [])
            rows.append({
                'image_id': image_id,
                'image_path': image_path,
                'caption': caption_text,
                'labels': labels
            })

    # Convert list of rows to dataframe
    df = pd.DataFrame(rows)
    return df


class COCOCaptionsDataset(Dataset):
    
    def __init__(self, df, device, img_transforms=True):
        self.df = df
        self.device = device
        self.transformations = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        if img_transforms:
            self.albumentations_transform = A.Compose([
                                                        A.Resize(height=256, width=256),
                                                        A.HorizontalFlip(p=0.5),
                                                        # A.VerticalFlip(p=0.5), 
                                                        A.Rotate(limit=30, p=0.5),
                                                        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                                                        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.5),
                                                    ])
        else:
            self.albumentations_transform = A.Compose([
                                                        A.Resize(height=256, width=256)
                                                    ])

            
        self.tokenizer, self.bert_model = init_bert_model()
        self.bert_model.to(f'cuda:{self.device}')
        
    def _load_img(self, img_path):
        
        self.img_path = img_path
        img = Image.open(self.img_path).convert('RGB')
        img = np.array(img)
        
        if self.albumentations_transform:
            img = self.albumentations_transform(image=img)["image"]
            
        to_tensor_transform = transforms.ToTensor()
        img = to_tensor_transform(img)
        
        if self.transformations:
            img = self.transformations(img)
            
        return img
        
    
    def __getitem__(self, idx):
        
        values = self.df.loc[idx, ['image_path', 'caption', 'image_label', 'caption_label', 'folder']].values
        
        img_path = values[0]
        caption = values[1]
        img_label = values[2]
        caption_label = values[3]
        folder = values[4]
        
        root_dir = r'/home/risaac6/Data/Atika/Model_Optimization/data/'
        img = self._load_img(os.path.join(root_dir, folder, img_path))
        inputs = self.tokenizer(caption, padding='max_length', max_length = 30, truncation=True, return_tensors="pt")
        inputs = {k:v.to(self.device) for k,v in inputs.items()}
        embeddings = self.bert_model(**inputs)
        embeddings = embeddings.pooler_output
        
        inputs = {}
        inputs['img'] = img
        inputs['embeddings'] = embeddings
        inputs['caption_label'] = caption_label
        inputs['caption'] = caption
        
        return inputs
        
    def __len__(self):
        return self.df.shape[0]
        
        
        
        
        
        