#
# coco.py
#

import os
import glob
import json
import numpy as np
import copy

import matplotlib.pyplot as plt
import cv2

class COCO:
    """
    读取/可视化 COCO 格式的标注

    + COCO_datasets
        + annotations
            - train.json
            - test.json
            - val.json
        + images
            + train
                - xxx.jpg
                - xxx.jpg
            + test
                - xxx.jpg
                - xxx.jpg
            + val
                - xxx.jpg
                - xxx.jpg
    """

    CLASSES = None

    def __init__(self, image_path=None,json_path=None,image_type="jpg"):
        
        self.labels = []
        self._init_label = {
            "info" : "",
            "licenses" : "",
            "images" : [],
            "annotations" : [],
            "categories" : [],
            
        }
        if image_path != None and json_path != None:
            self.CLASSES, self.labels = self._load_jsons_images(image_path, json_path, image_type)


    def __getitem__(self, idx):
        return self.get_label(idx)

    def __len__(self):
        return len(self.labels)

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

    def set_classes(self, classes):
            self.CLASSES = classes

    def get_classes(self):
        return self.CLASSES

    def get_label(self, index):
        label = copy.deepcopy(self.labels[index])
        return label

    def _load_jsons_images(self,image_path,json_path,image_type):
        with open(json_path) as fp:
            json_data = json.load(fp)
            # 'info', 'licenses', 'images', 'annotations', 'categories'
            images = json_data["images"]
            annotations = json_data["annotations"]
            categories = json_data["categories"]

            #
            id2image = {}
            for img in images:
                # ['license', 'id', 'file_name', 'width', 'height', 'coco_url', 'date_captured', 'flickr_url', 'season', 'weather', 'Day_Night', 'year', 'action']
                id2image[img["id"]] = {"name": os.path.join(image_path, img["file_name"]), 
                                        "width": img["width"],
                                        "height":img["height"],
                                        "bboxes": []}
            for anno in annotations:
                # ['supercategory', 'id', 'name', 'keypoints', 'skeleton']
                # ['id', 'image_id', 'category_id', 'bbox', 'area', 'iscrowd', 'num_keypoints', 'keypoints', 'segmentation']

                img_id = anno["image_id"]
                bbox = np.array(anno["bbox"], dtype=float)
                category_id = anno["category_id"]
                x1,y1,w,h = bbox
                cx = x1 + w/2
                cy = y1 + h/2
                H,W = id2image[img_id]["height"], id2image[img_id]["width"]

                cx /= W
                cy /= H
                w /= W
                h /= H
                id2image[img_id]["bboxes"].append([category_id, cx,cy,w,h])

            classes = {}
            for cates in categories:
                # ['supercategory', 'id', 'name', 'keypoints', 'skeleton']
                classes[cates["name"]] = cates["id"]

            labels = []
            for key in id2image.keys():
                labels.append({
                    "image": id2image[key]["name"],
                    "jsons": None,
                    "bboxes":id2image[key]["bboxes"]
                })
            return classes, labels
            
    def _vis_single(self, index, save_dir="."):
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
                self._vis_single(i)
        else:
            self._vis_single(index, save_dir=".")

# test
coco = COCO("/data/wildlife/coco_network/images/val2017/", "/data/wildlife/coco_network/annotations/wildlife_instance_val2017.json")
print(coco.get_label(100))
print(coco.get_classes())
coco.vis(100)

