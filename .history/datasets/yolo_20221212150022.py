#
# yolo.py
#

import os
import glob
import json
import numpy as np
import copy

import matplotlib.pyplot as plt
import cv2

class YOLO:
    """
    读取/可视化 YOLO 格式的标注

    + YOLO_datasets
        + images
            - xxx.jpg
            - xxx.jpg
        + labels
            - xxx.txt
            - xxx.txt
    """

    CLASSES = None

    def __init__(self, image_path=None,json_path=None,image_type="jpg"):
        
        self._size = 0
        if image_path != None and json_path != None:
            # 加载图片-标签文件
            self.label_dict = self._load_jsons_images(image_path, json_path, image_type)
            # 检查图片-标签文件是否存在，若不存在则删除
            self._check_label_image()
            
            # 图片-标签 数量
            self._size = len(self.label_dict.keys())

            # 加载标注文件数据
            self.CLASSES, self.labels = self._load_labels()
        else:
            pass

    def __getitem__(self, idx):
        return self.get_label(idx)

    def __len__(self):
        return self._size

    def _check_label_image(self):
        for json_file in self.label_dict.keys():
            image_file = self.label_dict[json_file]

            if not os.path.exists(json_file):
                print(json_file)
            if not os.path.exists(image_file):
                print(image_file)

    def _load_labels(self):
        classes = []
        bboxes = []
        for txt_file in self.label_dict.keys():
            image_file = self.label_dict[txt_file]

            label = []
            with open(txt_file) as fp:
                data = fp.readlines()
                for line in data:
                    row_data = line.strip().split(" ")
                    cls_id = int(row_data[0])
                    cx,cy,w,h = np.array(row_data[1:],dtype=float)
                    label.append([cls_id, cx, cy, w, h])
                    if cls_id not in classes:
                        classes.append(cls_id)
                bboxes.append({
                    "image": image_file,
                    "label": txt_file,
                    "bboxes": label,
                    })
            class_dict = None
            if isinstance(self.CLASSES, list):
                class_dict =  {name : idx for idx,name in enumerate(self.CLASSES)}
            if isinstance(self.CLASSES, dict):
                class_dict = self.CLASSES
            if self.CLASSES == None:
                class_dict = {c:i for i,c in enumerate(classes)}
        return class_dict, bboxes

    def get_classes(self):
        return self.CLASSES

    def get_label(self, index):
        label = copy.deepcopy(self.labels[index])
        for i in range(len(label["bboxes"])):
            label["bboxes"][i][0] = self.CLASSES[label["bboxes"][i][0]]
        return label

    def _load_jsons_images(self,image_path,json_path,image_type):
        json_data = {}
        json_files = glob.glob(os.path.join(json_path, "*.txt"))
        json_files.sort()
        for single_json in json_files:
            name = single_json.split("/")[-1].split(".")[0]
            image_name = image_path + "/" + name + "." + image_type
            json_data[single_json] = image_name
        return json_data

    def __vis_single(self, index, save_dir="."):
        # 可视化单张图片
        label = self.get_label(index)
        image = label["image"]
        bboxes = label["bboxes"]
        im = cv2.imread(image)
        H,W,_ = im.shape
        for bbox in bboxes:
            cls_id,cx,cy,w,h = bbox
            cx *= W
            cy *= H
            w *= W
            h *= H

            x1 = cx - w/2
            y1 = cy - h/2
            x2 = cx + w/2
            y2 = cy + h/2

            cv2.rectangle(im, (int(x1), int(y1)), (int(x2), int(y2)),(0,0,255),3)
        image_path = os.path.join(save_dir, image.split("/")[-1])
        cv2.imwrite(image_path, im)

    def vis(self, index=None,save_dir="."):
        if index == None:
            for i in range(self.__size,save_dir="."):
                self.__vis_single(i)
        else:
            self.__vis_single(index, save_dir=".")

# test
yolo = YOLO("/data/lj/datasets/em/yolo/images/trainval/", "/data/lj/datasets/em/yolo/labels/trainval/")
print(yolo.get_label(10))
print(yolo.get_classes())
yolo.vis(10)

