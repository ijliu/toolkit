#
# labelme.py
#

import os
import glob
import json
import numpy as np
import copy

import matplotlib.pyplot as plt
import cv2

class Labelme:
    """
    读取/可视化 Labelme 格式的标注
    """
    CLASSES = None

    def __init__(self, image_path=None,json_path=None,image_type="jpg"):

        # self.labels = [
            {
                "image": <image-name>
                "json" : <json-name>
                "bboxes" : [[cls,cx, cy, w, h], ... ]
            }
        ]
        self.labels = []
        
        if image_path != None and json_path != None:
            # 加载图片-标签文件
            self.label_dict = self._load_jsons_images(image_path, json_path, image_type)
            # 检查图片-标签文件是否存在，若不存在则删除
            self._check_label_image()

            # 加载标注文件数据
            self.CLASSES, self.labels = self._load_labels()
        else:
            pass

    def __getitem__(self, idx):
        return self.get_label(idx)

    def __len__(self):
        return  len(self.labels)

    def _check_label_image(self):
        for json_file in self.label_dict.keys():
            image_file = self.label_dict[json_file]

            if not os.path.exists(json_file):
                print(json_file)
            if not os.path.exists(image_file):
                print(image_file)

    def _load_labels(self):
        classes = []
        labels = []
        for json_file in self.label_dict.keys():
            image_file = self.label_dict[json_file]

            with open(json_file) as fp:
                data = json.load(fp)
                # 'version', 'flags', 'shapes', 'imageData', 'imagePath', 'imageHeight', 'imageWidth'
                shapes = data["shapes"]
                imageHeight = data["imageHeight"]
                imageWidth = data["imageWidth"]

                bboxes = []
                for shape in shapes:
                    # 'label', 'points', 'group_id', 'shape_type', 'flags'
                    cls_name = shape["label"]
                    if cls_name not in classes:
                        classes.append(cls_name)

                    points = np.array(shape["points"], dtype=float)
                    x = points[:,0]
                    y = points[:,1]

                    # bbox
                    minx = min(x)
                    maxx = max(x)
                    miny = min(y)
                    maxy = max(y)

                    # (x1,y1,x2,y2) -> (cx,cy,w,h)
                    cx = (minx + maxx)/2
                    cy = (miny + maxy)/2
                    w = maxx - minx
                    h = maxy - miny

                    # 归一化
                    cx /= imageWidth
                    cy /= imageHeight
                    w /= imageWidth
                    h /= imageHeight

                    bboxes.append([cls_name, cx, cy, w, h])
                labels.append({
                    "image": image_file,
                    "json": json_file,
                    "bboxes": bboxes,
                    })
            classes.sort()
            class_dict = None
            if isinstance(self.CLASSES, list):
                class_dict =  {name : idx for idx,name in enumerate(self.CLASSES)}
            if isinstance(self.CLASSES, dict):
                class_dict = self.CLASSES
            if self.CLASSES == None:
                class_dict = {c:i for i,c in enumerate(classes)}
        return class_dict, labels

    def get_classes(self):
        return self.CLASSES

    def get_label(self, index):
        label = copy.deepcopy(self.labels[index])
        for i in range(len(label["bboxes"])):
            label["bboxes"][i][0] = self.CLASSES[label["bboxes"][i][0]]
        return label

    def _load_jsons_images(self,image_path,json_path,image_type):
        json_data = {}
        json_files = glob.glob(os.path.join(json_path, "*.json"))
        json_files.sort()
        for single_json in json_files:
            name = single_json.split("/")[-1].split(".")[0]
            image_name = image_path + "/" + name + "." + image_type
            json_data[single_json] = image_name
        return json_data

    # 初始化
    def load(self, label):
        """

        Args:
            label : {
                "image": <image-name>,
                "json" : <json-name>,
                "bboxes" : [[cls_id, cx, cy, w, h]]
            }
        """
        
        pass
    
    def loads(self, labels):
        pass
    
    # 存储
    def dump(self, base_path):
        pass


    # 可视化
    def _vis_single(self, index, save_dir="."):
        # 可视化单张图片
        print(f"{index} VIS_SINGLE")
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
labelme = Labelme()
for i in range(10, 20):
    label = labelme.get_label(i)
    labelme.load(label)

print(len(labelme))
    
    
