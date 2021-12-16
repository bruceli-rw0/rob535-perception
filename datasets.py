#!/usr/bin/env python
"""
Dataloader for GTA car dataset. Load data in YOLOv5 format.

Author: Wai-Ting Li
Last modified: 1/1/2021
"""

import numpy as np
import pandas as pd
from torch import nn
from torch.utils.data import Dataset
from PIL import Image

from .utility import rot, get_bbox

class GTACarDataset(Dataset):

    def __init__(self, image_paths: list, class_file: str):
        super().__init__()

        self.image_paths = image_paths

        infos = [path[:-len('image.jpg')] + 'bbox.bin' for path in image_paths]
        projs = [path[:-len('image.jpg')] + 'proj.bin' for path in image_paths]

        classes = pd.read_csv(class_file)
        classId2label = {cid: label for cid, label in classes.iloc[:,[0,2]].to_numpy()}

        self.annotations = list()

        for i, (info, _proj) in enumerate(zip(infos, projs)):

            img = Image.open(self.image_paths[i])
            img_w, img_h = img.size

            bbox = np.fromfile(info, dtype=np.float32)
            proj = np.fromfile(_proj, dtype=np.float32)
            proj.resize([3, 4])

            R = rot(bbox[0:3])
            t = bbox[3:6]
            sz = bbox[6:9]
            cls = int(bbox[9])

            vert_3D, edges = get_bbox(-sz / 2, sz / 2)
            vert_3D = R @ vert_3D + t[:, np.newaxis]

            vert_2D = proj @ np.vstack([vert_3D, np.ones(vert_3D.shape[1])])
            vert_2D = vert_2D / vert_2D[2, :]

            x_min, x_max = np.min(vert_2D[0]), np.max(vert_2D[0])
            y_min, y_max = np.min(vert_2D[1]), np.max(vert_2D[1])
            x_min = 0 if x_min < 0 else x_min
            y_min = 0 if y_min < 0 else y_min
            x_max = img_w if x_max > img_w else x_max
            y_max = img_h if y_max > img_h else y_max

            width = x_max - x_min
            height = y_max - y_min
            x_center = x_min + width / 2
            y_center = y_min + height / 2
            
            # normalize coordinate w.r.t to image size
            width /= img_w
            height /= img_h
            x_center /= img_w
            y_center /= img_h

            self.annotations.append(
                [classId2label[cls], x_center, y_center, width, height]
            )

        assert len(self.image_paths) == len(self.annotations)

    def __getitem__(self, index):
        img = Image.open(self.image_paths[index])
        return img, self.annotations[index], self.image_paths[index]

    def __len__(self):
        return len(self.image_paths)
