import os
import torch
import random
import numpy as np
import pandas as pd
from PIL import Image

from torchvision import transforms
from torch.utils.data.dataset import Dataset

import albumentations as A
from albumentations.pytorch import ToTensorV2


# def get_cxr_image_captions_data(normal_class_id):
    
#     normal_class_id = int(normal_class_id[0])

# # if os.path.exists("cxr_image_captions.csv"):
#     # df_cxr_meta = pd.read_csv(r"cxr_image_captions.csv")
# # else:
#     # Read label file
#     root_dir = "../Data/risaac6/MIMIC_CXR_256/physionet.org/files/mimic-cxr-jpg/2.0.0/"
#     image_root = r"/home/risaac6/Data/risaac6/MIMIC_CXR_256/physionet.org/files/mimic-cxr-jpg/2.0.0/files/"
#     caption_root = r"/home/risaac6/Data/MIMIC_CXR_NOTES/files/"
#     file_name_chexpert = 'mimic-cxr-2.0.0-chexpert.csv'
#     file_name_meta = 'mimic-cxr-2.0.0-metadata.csv'

#     df = pd.read_csv(root_dir + file_name_chexpert)

#     # remove uncertain diagnoses
#     df = df[~(df == -1).any(axis = 1)]
#     df = df.fillna(0)

#     id_label_mapping = {0:'Atelectasis',                        
#                         1:'Cardiomegaly',
#                         2:'Consolidation',
#                         3:'Edema',
#                         4:'Enlarged Cardiomediastinum',
#                         5:'Fracture',
#                         6:'Lung Lesion',
#                         7:'Lung Opacity',
#                         8:'No Finding',
#                         9:'Pleural Effusion',
#                         10:'Pleural Other',
#                         11:'Pneumonia',
#                         12:'Pneumothorax',
#                         13:'Support Devices'}

#     df.loc[df[id_label_mapping[normal_class_id]] == 1, 'image_label'] = 0
#     df.loc[df[id_label_mapping[normal_class_id]] != 1, 'image_label'] = 1

#     # Read metadata file
#     df_cxr_meta = pd.read_csv(root_dir + file_name_meta)
#     # Filter for PA and merge with chexpert labels
#     df_cxr_meta = df_cxr_meta[df_cxr_meta['ViewPosition'] == 'PA']
#     df_cxr_meta = df_cxr_meta[["dicom_id", "subject_id", "study_id", "ViewPosition"]]
#     df_cxr_meta = df_cxr_meta.merge(df, on = ["subject_id", "study_id"], how = "inner")
#     # set paths
#     df_cxr_meta["image_path"] = image_root + 'p' + df_cxr_meta["subject_id"].astype(str).str[:2] + '/p' + df_cxr_meta["subject_id"].astype(str) \
#         + '/s'  + df_cxr_meta["study_id"].astype(str) + '/' + df_cxr_meta["dicom_id"] + '.jpg'
#     df_cxr_meta["caption_path"] = caption_root + 'p' + df_cxr_meta["subject_id"].astype(str).str[:2] + '/p' + df_cxr_meta["subject_id"].astype(str) \
#         + '/s' + df_cxr_meta["study_id"].astype(str) + ".txt"
#     df_cxr_meta["caption_label"] = 0   
#     # df_cxr_meta = df_cxr_meta[(df_cxr_meta["No Finding"] == 1.0) | (df_cxr_meta["Pneumothorax"] == 1.0)]
#     df_cxr_meta.to_csv("cxr_image_captions.csv", index = False)
    
#     return df_cxr_meta


### Pneumothorax ###
# def get_cxr_image_captions_data(normal_class_ids):
#     # Paths and filenames
#     root_dir = "../Data/risaac6/MIMIC_CXR_256/physionet.org/files/mimic-cxr-jpg/2.0.0/"
#     image_root = "/home/risaac6/Data/risaac6/MIMIC_CXR_256/physionet.org/files/mimic-cxr-jpg/2.0.0/files/"
#     caption_root = "/home/risaac6/Data/MIMIC_CXR_NOTES/files/"
#     file_name_chexpert = 'mimic-cxr-2.0.0-chexpert.csv'
#     file_name_meta = 'mimic-cxr-2.0.0-metadata.csv'

#     # Read Chexpert label file
#     df = pd.read_csv(root_dir + file_name_chexpert)

#     # Remove uncertain diagnoses
#     df = df[~(df == -1).any(axis=1)]
#     df = df.fillna(0)

#     # Mapping of ID to labels
#     id_label_mapping = {
#         0: 'Atelectasis',
#         1: 'Cardiomegaly',
#         2: 'Consolidation',
#         3: 'Edema',
#         4: 'Enlarged Cardiomediastinum',
#         5: 'Fracture',
#         6: 'Lung Lesion',
#         7: 'Lung Opacity',
#         8: 'No Finding',
#         9: 'Pleural Effusion',
#         10: 'Pleural Other',
#         11: 'Pneumonia',
#         12: 'Pneumothorax',
#         13: 'Support Devices'
#     }

#     # Apply the class filter for given IDs
#     df['image_label'] = 1  # Default to 1 (abnormal)
#     for class_id in normal_class_ids:
#         label = id_label_mapping[class_id]
#         df.loc[df[label] == 1, 'image_label'] = 0  # Set to 0 (normal) if condition is met

#     # Read metadata file
#     df_cxr_meta = pd.read_csv(root_dir + file_name_meta)

#     # Filter for PA view and merge with Chexpert labels
#     df_cxr_meta = df_cxr_meta[df_cxr_meta['ViewPosition'] == 'PA']
#     df_cxr_meta = df_cxr_meta[["dicom_id", "subject_id", "study_id", "ViewPosition"]]
#     df_cxr_meta = df_cxr_meta.merge(df, on=["subject_id", "study_id"], how="inner")

#     # Set paths
#     df_cxr_meta["image_path"] = image_root + 'p' + df_cxr_meta["subject_id"].astype(str).str[:2] + '/p' + df_cxr_meta["subject_id"].astype(str) \
#         + '/s' + df_cxr_meta["study_id"].astype(str) + '/' + df_cxr_meta["dicom_id"] + '.jpg'
#     df_cxr_meta["caption_path"] = caption_root + 'p' + df_cxr_meta["subject_id"].astype(str).str[:2] + '/p' + df_cxr_meta["subject_id"].astype(str) \
#         + '/s' + df_cxr_meta["study_id"].astype(str) + ".txt"
#     df_cxr_meta["caption_label"] = 0

#     # Save to CSV file
#     df_cxr_meta.loc[(df_cxr_meta["Pneumothorax"] == 1) | (df_cxr_meta["No Finding"] == 1)]
#     df_cxr_meta.to_csv("cxr_image_captions.csv", index=False)
    
#     return df_cxr_meta

def get_cxr_image_data(normal_class_ids):
    # Paths and filenames
    root_dir = "../Data/risaac6/MIMIC_CXR_256/physionet.org/files/mimic-cxr-jpg/2.0.0/"
    image_root = "/home/risaac6/Data/risaac6/MIMIC_CXR_256/physionet.org/files/mimic-cxr-jpg/2.0.0/files/"
    caption_root = "/home/risaac6/Data/MIMIC_CXR_NOTES/files/"
    file_name_chexpert = 'mimic-cxr-2.0.0-chexpert.csv'
    file_name_meta = 'mimic-cxr-2.0.0-metadata.csv'

    # Read Chexpert label file
    df = pd.read_csv(root_dir + file_name_chexpert)

    # Remove uncertain diagnoses
    df = df[~(df == -1).any(axis=1)]
    df = df.fillna(0)

    # Mapping of ID to labels
    id_label_mapping = {
        0: 'Atelectasis',
        1: 'Cardiomegaly',
        2: 'Consolidation',
        3: 'Edema',
        4: 'Enlarged Cardiomediastinum',
        5: 'Fracture',
        6: 'Lung Lesion',
        7: 'Lung Opacity',
        8: 'No Finding',
        9: 'Pleural Effusion',
        10: 'Pleural Other',
        11: 'Pneumonia',
        12: 'Pneumothorax',
        13: 'Support Devices'
    }

    # Apply the class filter for given IDs
    df['image_label'] = 1  # Default to 1 (abnormal)
    for class_id in normal_class_ids:
        label = id_label_mapping[class_id]
        df.loc[df[label] == 1, 'image_label'] = 0  # Set to 0 (normal) if condition is met

    # Read metadata file
    df_cxr_meta = pd.read_csv(root_dir + file_name_meta)

    # Filter for PA view and merge with Chexpert labels
    df_cxr_meta = df_cxr_meta[df_cxr_meta['ViewPosition'] == 'PA']
    df_cxr_meta = df_cxr_meta[["dicom_id", "subject_id", "study_id", "ViewPosition"]]
    df_cxr_meta = df_cxr_meta.merge(df, on=["subject_id", "study_id"], how="inner")

    # Set paths
    df_cxr_meta["image_path"] = image_root + 'p' + df_cxr_meta["subject_id"].astype(str).str[:2] + '/p' + df_cxr_meta["subject_id"].astype(str) \
        + '/s' + df_cxr_meta["study_id"].astype(str) + '/' + df_cxr_meta["dicom_id"] + '.jpg'
    df_cxr_meta["caption_path"] = caption_root + 'p' + df_cxr_meta["subject_id"].astype(str).str[:2] + '/p' + df_cxr_meta["subject_id"].astype(str) \
        + '/s' + df_cxr_meta["study_id"].astype(str) + ".txt"
    df_cxr_meta["caption_label"] = 0

    # Save to CSV file
    df_cxr_meta.to_csv("cxr_image_captions.csv", index=False)
    
    return df_cxr_meta

def get_cxr_data(normal_class_id):
    
    df = get_cxr_image_data(normal_class_id)
    df = df[["image_path", "image_label"]]
    
    normal_filenames = df[df["image_label"] == 0]["image_path"].values.tolist()
    outlier_filenames = df[df["image_label"] == 1]["image_path"].values.tolist()
    
    return normal_filenames, outlier_filenames
    

class CXR_Dataset(Dataset):
    
    def __init__(self, imagenames, labels, img_transforms=True):
        
        self.imagenames = imagenames
        self.labels = labels
        
        self.transformations = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        if img_transforms:
            self.albumentations_transform = A.Compose([
                                                        A.Resize(height=256, width=256),
                                                        # A.HorizontalFlip(p=0.5),
                                                        # A.VerticalFlip(p=0.5), 
                                                        # A.Rotate(limit=30, p=0.5),
                                                        # A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                                                        # A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.5),
                                                    ])
        else:
            self.albumentations_transform = A.Compose([
                                                        A.Resize(height=256, width=256)
                                                    ])
        
        
    def __load_image(self, idx):
        
        img = Image.open(self.imagenames[idx]).convert("RGB")
        img = np.array(img)
        if self.albumentations_transform:
            img = self.albumentations_transform(image=img)["image"]
        
        to_tensor_transform = transforms.ToTensor()
        img = to_tensor_transform(img)
        
        if self.transformations:
            img = self.transformations(img)
        
        return img, self.labels[idx]
                
    def __getitem__(self, idx):        
        return self.__load_image(idx)

        
    def __len__(self):
        return len(self.imagenames)
        
        
def get_cxr_image_captions_data(normal_class_ids):
    # Paths and filenames
    root_dir = "../Data/risaac6/MIMIC_CXR_256/physionet.org/files/mimic-cxr-jpg/2.0.0/"
    image_root = "/home/risaac6/Data/risaac6/MIMIC_CXR_256/physionet.org/files/mimic-cxr-jpg/2.0.0/files/"
    caption_root = "/home/risaac6/Data/MIMIC_CXR_NOTES/files/"
    file_name_chexpert = 'mimic-cxr-2.0.0-chexpert.csv'
    file_name_meta = 'mimic-cxr-2.0.0-metadata.csv'

    # Read Chexpert label file
    df = pd.read_csv(root_dir + file_name_chexpert)

    # Remove uncertain diagnoses
    df = df[~(df == -1).any(axis=1)]
    df = df.fillna(0)

    # Mapping of ID to labels
    id_label_mapping = {
        0: 'Atelectasis',
        1: 'Cardiomegaly',
        2: 'Consolidation',
        3: 'Edema',
        4: 'Enlarged Cardiomediastinum',
        5: 'Fracture',
        6: 'Lung Lesion',
        7: 'Lung Opacity',
        8: 'No Finding',
        9: 'Pleural Effusion',
        10: 'Pleural Other',
        11: 'Pneumonia',
        12: 'Pneumothorax',
        13: 'Support Devices'
    }

    # Apply the class filter for given IDs
    df['image_label'] = 1  # Default to 1 (abnormal)
    for class_id in normal_class_ids:
        label = id_label_mapping[class_id]
        df.loc[df[label] == 1, 'image_label'] = 0  # Set to 0 (normal) if condition is met

    # Read metadata file
    df_cxr_meta = pd.read_csv(root_dir + file_name_meta)

    # Filter for PA view and merge with Chexpert labels
    df_cxr_meta = df_cxr_meta[df_cxr_meta['ViewPosition'] == 'PA']
    df_cxr_meta = df_cxr_meta[["dicom_id", "subject_id", "study_id", "ViewPosition"]]
    df_cxr_meta = df_cxr_meta.merge(df, on=["subject_id", "study_id"], how="inner")

    # Set paths
    df_cxr_meta["image_path"] = image_root + 'p' + df_cxr_meta["subject_id"].astype(str).str[:2] + '/p' + df_cxr_meta["subject_id"].astype(str) \
        + '/s' + df_cxr_meta["study_id"].astype(str) + '/' + df_cxr_meta["dicom_id"] + '.jpg'
    df_cxr_meta["caption_path"] = caption_root + 'p' + df_cxr_meta["subject_id"].astype(str).str[:2] + '/p' + df_cxr_meta["subject_id"].astype(str) \
        + '/s' + df_cxr_meta["study_id"].astype(str) + ".txt"
    
    # Specify columns to match on
    columns_to_match = ['Column1', 'Column2']

    # Merge with an indicator, only on specified columns
    merged_df = df_cxr_meta.merge(df2[columns_to_match], on=columns_to_match, how='left', indicator=True)

    # Filter out rows that are found in both dataframes based on specified columns
    result_df = merged_df[merged_df['_merge'] == 'left_only'].drop(columns=['_merge'])


    # Save to CSV file
    df_cxr_meta.to_csv("cxr_image_captions.csv", index=False)
    
    return df_cxr_meta


        
class CXRCaptionsDataset(Dataset):
    
    def __init__(self, df, device, img_transforms=True):
        self.df = df
        self.device = device
        self.transformations = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        if img_transforms:
            self.albumentations_transform = A.Compose([
                                                        A.Resize(height=256, width=256),
                                                        # A.HorizontalFlip(p=0.5),
                                                        # A.VerticalFlip(p=0.5), 
                                                        # A.Rotate(limit=30, p=0.5),
                                                        # A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                                                        # A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.5),
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
        