import json
import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import glob
from torch.utils.data import Dataset
from sklearn import preprocessing
import torch
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

data_root = '/media/hxd/82231ee6-d2b3-4b78-b3b4-69033720d8a8/MyDatasets/amazon'
version = 'img_by_attr_50'

# load product
products = []
json_files = glob.glob(data_root + '/metadata/*.json')
for json_file in json_files:
    for line in open(json_file, 'r'):
        products.append(json.loads(line))

def load_data(attr, addfolder = ''):
    cur_attr_vals = next(os.walk(data_root + '/' + version + '/' + attr))[1]

    cur_attr_img_paths = []
    attrs_labels = []
    for val in cur_attr_vals:
        tmp = glob.glob(data_root + '/' + version + '/' + attr + '/' + val + '/*.jpg')
        cur_attr_img_paths.extend(tmp)
        attrs_labels.extend([val] * len(tmp))

    if addfolder:
        add_attr_vals = next(os.walk(data_root + '/' + version + '/' + addfolder))[1]
        add_attr_img_paths = []
        for val in add_attr_vals:
            add_attr_img_paths.extend(glob.glob(data_root + '/' + version + '/' + addfolder + '/' + val + '/*.jpg'))
        
        add_img_paths = []
        add_labels = []
        for img_path in add_attr_img_paths:
            id = img_path.split('/')[-1][:-4]
            record = next(item for item in products if 'main_image_id' in item.keys() and item['main_image_id'] == id and len([x for x in item['item_name'] if x['language_tag']=='en_US'])>0)

            if attr in record.keys():
                att_value = [x['value'] for x in record[attr] if x['language_tag'] == 'en_US']
                if len(att_value) > 0 and att_value[0].lower() in cur_attr_vals:
                    add_labels.extend([att_value[0].lower()])
                    add_img_paths.extend([img_path])
        
        cur_attr_img_paths.extend(add_img_paths)
        attrs_labels.extend(add_labels)

    return cur_attr_img_paths, attrs_labels            


class Dataset(Dataset):

    def __init__(self, images, le, labels, data_indx):
        self.images = images
        self.labels = le.transform(labels)
        self.data_indx = data_indx

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {'image': self.images[idx], 
                  'label': self.labels[idx],
                  'data_indx': self.data_indx[idx]
                 }

        return sample
   