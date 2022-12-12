#
# labelme.py
#

import os
import glob
import json
import numpy as np
import copy
import shutil

import matplotlib.pyplot as plt
import cv2

def convert(init_label, label, classes):
    init_label["imagePath"] = os.path.split(label["image"])[-1]
    init_label["imageHeight"], init_label["imageWidth"], _ = cv2.imread(label["image"]).shape
    
    for box in label["bboxes"]:
        shape = {}
        cls_id,cx,cy,w,h = box
        
        if classes == None:
            classes = {}
        if cls_id not in classes:
            classes[str(cls_id)] = cls_id

        id2classes = {classes[key] : key for key in classes.keys()}
        
        w *= init_label["imageWidth"]
        h *= init_label["imageHeight"]
        cx *= w
        cy *= h
        
        x1 = cx - w/2
        y1 = cy - h/2
        x2 = cx + w/2
        y2 = cy + h/2
        
        shape = {
            "label" : id2classes[cls_id],
            "points" : [[x1,y1], [x2,y2]],
            "group_id" : None,
            "shape_type" : "rectangle",
            "flags" : {}
        }
        init_label["shapes"].append(shape)
    return init_label, classes


class Labelme:
    """
    读取/可视化 Labelme 格式的标注
    """
    CLASSES = None

    def __init__(self, image_path=None,json_path=None,image_type="jpg"):

        '''
        self.labels = [
            {
                "image": <image-name>
                "json" : <json-name>
                "bboxes" : [[cls_id,cx, cy, w, h], ... ]
            }
        ]
        '''
        self.labels = []
        self.__init_label = {
                "version": "5.1.1",
                "flags": {},
                "shapes":[],
                "imageData": None,
                "imagePath" : "",
                "imageHeight": 0,
                "imageWidth": 0,
            }
        
        if image_path != None and json_path != None:
            # 加载图片-标签文件
            self.label_dict = self._load_jsons_images(image_path, json_path, image_type)
            # 检查图片-标签文件是否存在，若不存在则删除
            self._check_label_image()
            # 加载标注文件数据
            self.CLASSES, self.labels = self._load_labels()

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
    
    def set_classes(self, classes):
        self.CLASSES = classes

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
    def load(self, label, func = convert):
        """
        Args:
            label : {
                "image": <image-name>,
                "json" : <json-name>,
                "bboxes" : [[cls_id, cx, cy, w, h]]
            }
        """
        self.labels.append(label)
    
    def loads(self, labels):
        for label in labels:
            self.load(label)
        pass
    
    # 存储
    def save(self, base_path, func = convert):
        if not os.path.exists(base_path):
            os.makedirs(base_path)
            
        for label in self.labels:
            init_label = self.__init_label
            # 设置转换函数，可以将label转换为目标格式
            json_label, self.CLASSES = func(init_label, label, self.CLASSES)
            
            file_name = os.path.split(label["image"])[-1].split(".")[0] + ".json"
            
            with open(os.path.join(base_path, file_name), 'w') as fp:
                json.dump(json_label, fp, indent=4)
                
            # 是否复制图像文件
            shutil.copyfile(label["image"], os.path.join(base_path, os.path.split(label["image"])[-1]))

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
l1 = Labelme("/data/lj/datasets/new_pandas/", "/data/lj/datasets/new_pandas/")
l1_cls = l1.get_classes()
print(l1_cls)

l2 = Labelme()
l2.set_classes(l1_cls)

print(len(l1))
print(len(l2))
for i in range(10, 20):
    label = l1.get_label(i)
    l2.load(label)
    
l2.save("labelme/")

print(len(l1))
print(len(l2))
    
    
