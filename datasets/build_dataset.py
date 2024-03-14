import logging
import numpy as np

from .SIIM import *
from .CIFAR10Dataset import *
from .COCO import get_coco_data, COCO_Dataset, get_coco_image_captions_data, COCOCaptionsDataset

from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.distributed import DistributedSampler
from torch import nn


cifar10_tsfms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

def build_cvae_dataset(dataset_name, data_path, cvae_batch_size, normal_class):
    logger = logging.getLogger()
    logger.info("Build CVAE dataset for {}".format(dataset_name))
    
    assert dataset_name in ['cifar10', 'siim', 'coco']
    
    if dataset_name == "cifar10":
      
        normal_x_train, normal_y_train, normal_x_val, normal_y_val, outlier_x_test, outlier_y_test = get_cifar10_data(normal_class)

        train_set = CIFAR10LabelDataset(normal_x_train, normal_y_train, cifar10_tsfms)
        validate_set = CIFAR10LabelDataset(normal_x_val, normal_y_val, cifar10_tsfms)
        test_set = CIFAR10LabelDataset(normal_x_val+outlier_x_test, normal_y_val+outlier_y_test, cifar10_tsfms)
    
    elif dataset_name == "siim":
        
        benign, malignant = get_siim_data()
        
        train_set = SIIM_Dataset("./data/SIIM/train/", benign[0:int(0.8*(len( benign)))], [0]*len(benign[0:int(0.8*(len( benign)))]))
        validate_set = SIIM_Dataset("./data/SIIM/train/", benign[int(0.8*(len( benign)))+1:], [0]*len(benign[int(0.8*(len( benign)))+1:]))
        test_set = SIIM_Dataset("./data/SIIM/train/", benign[int(0.8*(len( benign)))+1:]+malignant, [0]*len(benign[int(0.8*(len( benign)))+1:])+[1]*len(malignant))

    elif dataset_name == 'coco':

        normal_filenames, outlier_filenames = get_coco_data(normal_class)
        print("Normal :",len(normal_filenames), " Outliers :", len(outlier_filenames))
        train_set = COCO_Dataset("../Data/Atika/Model_Optimization/data/train2017/", normal_filenames[0:int(0.8*(len(normal_filenames)))], [0]*len(normal_filenames[0:int(0.8*(len(normal_filenames)))]))
        validate_set = COCO_Dataset("../Data/Atika/Model_Optimization/data/train2017/", normal_filenames[int(0.8*(len( normal_filenames)))+1:], [0]*len(normal_filenames[int(0.8*(len( normal_filenames)))+1:]))
        test_set = COCO_Dataset("../Data/Atika/Model_Optimization/data/train2017/", normal_filenames[int(0.8*(len( normal_filenames)))+1:]+outlier_filenames, [0]*len(normal_filenames[int(0.8*(len( normal_filenames)))+1:])+[1]*len(outlier_filenames))
        
        # train_set = COCO_Dataset("../Data/Atika/Model_Optimization/data/train2017/", normal_filenames[0:int(0.1*(len(normal_filenames)))], [0]*len(normal_filenames[0:int(0.1*(len(normal_filenames)))]))
        # validate_set = COCO_Dataset("../Data/Atika/Model_Optimization/data/train2017/", normal_filenames[int(0.9*(len( normal_filenames)))+1:], [0]*len(normal_fil
        
        # enames[int(0.9*(len( normal_filenames)))+1:]))
        # test_set = COCO_Dataset("../Data/Atika/Model_Optimization/data/train2017/", normal_filenames[int(0.9*(len( normal_filenames)))+1:]+outlier_filenames, [0]*len(normal_filenames[int(0.9*(len( normal_filenames)))+1:])+[1]*len(outlier_filenames))
        
        print("Train :", len(train_set), "Validate :", len(validate_set), "Test :", len(test_set))
        
#     elif dataset_name == 'cxr':
        
#         normal_filenames, outlier_filenames = get_cxr_data(normal_class)

    cvae_dataloaders = {'train': DataLoader(train_set, batch_size = cvae_batch_size, shuffle = False, sampler=DistributedSampler(train_set), num_workers = 8),
                      'val': DataLoader(validate_set, batch_size = cvae_batch_size, sampler=DistributedSampler(validate_set), num_workers = 8),
                      'test': DataLoader(test_set, batch_size = cvae_batch_size, sampler=DistributedSampler(test_set), num_workers = 8)}
    cvae_dataset_sizes = {'train': len(train_set), 'val': len(validate_set), 'test':len(test_set)}        
        
        
    return cvae_dataloaders, cvae_dataset_sizes


def build_efnet_dataset(dataset_name, efnet_batch_size, normal_class, device):
    logger = logging.getLogger()
    logger.info("Build EFNET dataset for {}".format(dataset_name))
    
    assert dataset_name in ['coco']
    
    if dataset_name == 'coco':
        
        df = get_coco_image_captions_data(normal_class)
        
        # Calculate the split indices
        split_idx1 = int(len(df) * 0.7)
        split_idx2 = int(len(df) * 0.9)
        split_idx3 = int(len(df) * 1)
        
        train_set = COCOCaptionsDataset(df.iloc[:split_idx1].reset_index(drop=True), device)
        validate_set = COCOCaptionsDataset(df.iloc[split_idx1:split_idx2].reset_index(drop=True), device)
        test_set = COCOCaptionsDataset(df.iloc[split_idx2:split_idx3].reset_index(drop=True), device, img_transforms=False)
        
    efnet_dataloaders = {
                         'train': DataLoader(train_set, batch_size = efnet_batch_size, shuffle = False, sampler=DistributedSampler(train_set), num_workers = 0),
                         'val': DataLoader(validate_set, batch_size = efnet_batch_size, shuffle = False, sampler=DistributedSampler(validate_set), num_workers = 0),
                         'test': DataLoader(test_set, batch_size = efnet_batch_size, shuffle = False, sampler=DistributedSampler(test_set), num_workers = 0)
                        }
    efnet_dataset_sizes = {'train': len(train_set), 'val': len(validate_set), 'test': len(test_set)}
    
    return efnet_dataloaders, efnet_dataset_sizes