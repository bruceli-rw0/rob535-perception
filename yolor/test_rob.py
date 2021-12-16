import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm.auto import tqdm

from utils.google_utils import attempt_load
from utils.datasets import create_dataloader
from utils.general import coco80_to_coco91_class, check_dataset, check_file, check_img_size, box_iou, \
    non_max_suppression, scale_coords, xyxy2xywh, xywh2xyxy, clip_coords, set_logging, increment_path
from utils.loss import compute_loss
from utils.metrics import ap_per_class
from utils.plots import plot_images, output_to_target
from utils.torch_utils import select_device, time_synchronized

from models.models import *

def test(
    data,
    weights=None,
    batch_size=16,
    imgsz=640,
    conf_thres=0.001,
    iou_thres=0.6,  # for NMS
    save_json=False,
    single_cls=False,
    augment=False,
    verbose=False,
    model=None,
    dataloader=None,
    save_dir=Path(''),  # for saving images
    save_txt=False,  # for auto-labelling
    save_conf=False,
    plots=True,
    log_imgs=0, # number of logged images
):

    # Initialize/load model and set device
    training = model is not None
    device = next(model.parameters()).device  # get model device

    # Half
    half = device.type != 'cpu'  # half precision only supported on CUDA
    if half:
        model.half()

    # Configure
    model.eval()
    with open(data) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)  # model dict
    check_dataset(data)  # check
    nc = 1 if single_cls else int(data['nc'])  # number of classes
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    # Logging
    log_imgs, wandb = min(log_imgs, 100), None  # ceil
    try:
        import wandb  # Weights & Biases
    except ImportError:
        log_imgs = 0

    s = ('%20s' + '%12s' * 6) % ('Class', 'Images', 'Targets', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    p, r, f1, mp, mr, map50, map, t0, t1 = 0., 0., 0., 0., 0., 0., 0., 0., 0.
    loss = torch.zeros(3, device=device)
    jdict, stats, ap, ap_class, wandb_images = [], [], [], [], []

    pred_correct = 0
    tot_img = 0
    for batch_i, (img, targets, paths, shapes) in enumerate(tqdm(dataloader, desc=s)):
        img = img.to(device, non_blocking=True)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        targets = targets.to(device)
        nb, _, height, width = img.shape  # batch size, channels, height, width
        whwh = torch.Tensor([width, height, width, height]).to(device)

        # Disable gradients
        with torch.no_grad():
            # Run model
            t = time_synchronized()
            inf_out, _ = model(img, augment=augment)  # inference and training outputs
            t0 += time_synchronized() - t

            # Run NMS
            t = time_synchronized()
            output = non_max_suppression(inf_out, conf_thres=conf_thres, iou_thres=iou_thres)
            t1 += time_synchronized() - t

        # Statistics per image
        for si, pred in enumerate(output):
            label = targets[targets[:, 0] == si, 1:]
            target_class = label[0,0]

            if len(pred):
                max_conf_pred = torch.argmax(pred[:,4]).item()
                pred_class = pred[max_conf_pred, 5]
            else:
                # no object detected, set pred to 0
                pred_class = 0

            assert pred_class < 3

            tot_img += 1
            if target_class == pred_class:
                pred_correct += 1

            # breakpoint()

    return pred_correct / tot_img, None, t


if __name__ == '__main__':
    pass
