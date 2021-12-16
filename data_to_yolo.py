import os
import shutil
from glob import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from datasets import GTACarDataset
from utility import visualize_bbox_yolo

def trainval():
    files = sorted(glob('data/trainval/*/*_image.jpg'))
    dataset = GTACarDataset(files, 'data/classes.csv')
    
    train_idx, val_idx = train_test_split(range(len(dataset)), test_size=0.2)

    dst_root = 'classifier/datasets/gtacar'

    os.mkdir(dst_root)
    os.mkdir(f'{dst_root}/images')
    os.mkdir(f'{dst_root}/labels')
    os.mkdir(f'{dst_root}/images/train')
    os.mkdir(f'{dst_root}/images/val')
    os.mkdir(f'{dst_root}/labels/train')
    os.mkdir(f'{dst_root}/labels/val')

    for indexes, _type_ in zip([train_idx, val_idx], ['train', 'val']):
        for i in indexes[:3]:
            _, annotation, src_img = dataset[i]
            path_split = src_img.split('/')
            iid = path_split[-1][:-len('_image.jpg')]
            dst_img = os.path.join(dst_root, 'images', _type_, f"{path_split[-2]}-{iid}.jpg")
            shutil.copyfile(src_img, dst_img)

            dst_label = os.path.join(dst_root, 'labels', _type_, f"{path_split[-2]}-{iid}.txt")
            c, x, y, w, h = annotation
            with open(dst_label, 'w') as f:
                f.write(f"{c} {x} {y} {w} {h}\n")

def test():
    files = sorted(glob('data/test/*/*_image.jpg'))
    dst_root = 'classifier/datasets/gtacar/images/test'
    for src_img in files:
        path_split = src_img.split('/')
        iid = path_split[-1][:-len('_image.jpg')]
        dst_img = os.path.join(dst_root, f"{path_split[-2]}.{iid}.jpg")
        shutil.copyfile(src_img, dst_img)

if __name__ == "__main__":
    trainval()
    test()
