import os
import cv2
import json
import pathlib
import numpy as np
import pandas as pd
from utils import get_config_yaml
from matplotlib import pyplot as plt
from dataset import read_img

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

def class_balance_check(patchify, data_dir):
    if patchify:
        with open(data_dir, 'r') as j:
            train_data = json.loads(j.read())
        labels = train_data['masks']
        patch_idx = train_data['patch_idx']
    else:
        train_data = pd.read_csv(data_dir)
        labels = train_data.masks.values
        patch_idx = None
    
    total = 0
    class_name = {}

    for i in range(len(labels)):
        mask = cv2.imread(labels[i])
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        if patchify:
            idx = patch_idx[i]
            mask = mask[idx[0]:idx[1], idx[2]:idx[3]]
            
        total_pix = mask.shape[0]*mask.shape[1]
        total += total_pix
        
        dic = {}
        keys = np.unique(mask)
        for i in keys:
            dic[i] = np.count_nonzero(mask == i)
            
        for key, value in dic.items(): 
            if key in class_name.keys():
                class_name[key] = value + class_name[key]
            else:
                class_name[key] = value
        
    for key, val in class_name.items():
        class_name[key] = (val/total)*100
        
    print("Class percentage:")
    for key, val in class_name.items():
        print("class pixel: {} = {}".format(key, val))
    
def check_height_width(data_dir):
    data = pd.read_csv(data_dir)
    # removing UU or UMM or UM
    # data = data[data['feature_ids'].str.contains('uu_00') == False]
    data = data[data['feature_ids'].str.contains('umm_00') == False]
    data = data[data['feature_ids'].str.contains('um_00') == False]
    
    print("Dataset:  ", data.shape)
    
    input_img = data.feature_ids.values
    input_mask = data.masks.values
    
    input_img_shape = []
    input_mask_shape = []
    
    for i in range(len(input_img)):
        img = cv2.imread(input_img[i])
        mask = cv2.imread(input_mask[i])

        if img.shape not in input_img_shape:
            input_img_shape.append(img.shape)

        if mask.shape not in input_mask_shape:
            input_mask_shape.append(mask.shape)

    print("Input image shapes: ", input_img_shape)
    print("Input mask shapes: ", input_mask_shape)


def display_all(data):
    pathlib.Path((config['visualization_dir']+'display')).mkdir(parents = True, exist_ok = True)

    for i in range(len(data)):
        image = read_img(data.feature_ids.values[i])
        mask = read_img(data.masks.values[i], label=True)
        id = data.feature_ids.values[i].split("/")[-1]
        display_list = {
                     "image":image,
                     "label":mask}

        plt.figure(figsize=(12, 8))
        title = list(display_list.keys())

        for i in range(len(display_list)):
            plt.subplot(1, len(display_list), i+1)
            plt.title(title[i])
            plt.imshow((display_list[title[i]]), cmap='gray')
            plt.axis('off')

        prediction_name = "img_id_{}.png".format(id) # create file name to save
        plt.savefig(os.path.join((config['visualization_dir']+'display'), prediction_name), bbox_inches='tight', dpi=800)
        plt.clf()
        plt.cla()
        plt.close()
