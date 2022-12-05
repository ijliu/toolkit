#
# labelme.py
#

import os
import glob
import json
import numpy as np

import matplotlib.pyplot as plt
import cv2

class Labelme:
	"""
	读取/可视化 Labelme 格式的标注
	"""

	CLASS = None

	def __init__(self, image_path,json_path,image_type=".jpg"):
		# 加载图片文件
		self.images = self.__load_images(image_path, image_type)
		# 加载json文件
		self.jsons = self.__load_jsons(json_path)

		self.classes = self.__get_classes()

		for (img, js) in zip(self.images, self.jsons):
			print(img)
			print(js)
			print("---------")
		pass

	def __load_images(self,image_path, image_type):
		images = glob.glob(os.path.join(image_path, "*"+image_type))
		images.sort()
		return images
	
	def __load_jsons(self,json_path):
		jsons = glob.glob(os.path.join(json_path, "*.json"))
		jsons.sort()
		return jsons

	def __get_classes(self):
		classes = None
		if isinstance(self.CLASS, list):
			return {name : idx for idx,name in enumerate(self.CLASS)}
		if isinstance(self.CLASS, dict):
			return self.CLASS

		classes = []
		for label in self.jsons:
			with open(label) as fp:
				data = json.load(fp)
				print(data.keys())
				exit()


		return classes
	
	def vis(self,):
		# 可视化
		pass

lm = Labelme("/home/jing/new_pandas/", "/home/jing/new_pandas/")

