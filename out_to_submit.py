"""
This scripts convert the YOLO output to the submission format
"""
import argparse
import os
import time
from glob import glob

def main(model, target_folder):
    root = f'classifier/{model}/runs/detect/{target_folder}'
    label_path = os.path.join(root, 'labels')
    files = glob(os.path.join(label_path, '*.txt'))

    assert len(files) <= 2651

    img_folders = glob('data/test/*')
    img_folder = img_folders[0]
    images = glob(f'{img_folder}/*_image.jpg')
    label_path = f'{root}/labels/*.txt'
    label_files = glob(label_path)

    timeID = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

    with open(f'submission/{timeID}.csv', 'w') as outf:
        outf.write('guid/image,label\n')
        
        for img_folder in img_folders:
            images = glob(f'{img_folder}/*_image.jpg')
            folder_name = img_folder.split('/')[-1]
            for image in images:
                img_id = image.split('/')[-1][:-len('_image.jpg')]

                label_file = f"{root}/labels/{folder_name}.{img_id}.txt"
                if label_file in label_files:
                    with open(label_file, 'r') as f:
                        content = f.read().strip()
                        preds = content.split('\n')
                        best_conf = 0
                        best_label = 0
                        for pred in preds:
                            label, _, _, _, _, conf = pred.strip().split(' ')
                            conf = float(conf)
                            if conf > best_conf:
                                best_conf = conf
                                best_label = int(label)
                        outf.write(f'{folder_name}/{img_id},{best_label}\n')
                else:
                    outf.write(f'{folder_name}/{img_id},0\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, type=str, default='', help='the model name', choices=['yolov5', 'yolor'])
    parser.add_argument('--folder', required=True, type=str, default='', help='the output folder from detection')
    opt = parser.parse_args()

    main(opt.model, opt.folder)
