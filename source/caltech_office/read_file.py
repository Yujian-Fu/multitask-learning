# -*- coding: utf-8 -*- 


import numpy as np
from PIL import Image
import math

def read_image(path):
	f = open(path)#/home/fyj/桌面/Multitask_learning/MTlearn-master/data/office-home/Art/test_5_.txt
	line = f.readline()
	print(line)
	label = line.split(" ")[1].split("\n")[0]

	print(type(label))
	line = line.split(" ")[0]
	print (line)


	image_raw = Image.open(line)   #bytes
	image_raw = image_raw.convert('L')

	result_x = image_raw
	result_y = []
	result_y.append(int(label))


	while line!= '':
		line = f.readline()
		print(line)
		if line.split('\n') != ['']:
			label = int(line.split(" ")[1].split("\n")[0])
			print(label)
			line = line.split(" ")[0]
			print (line)
			img = Image.open(line)
			img = img.convert('L')
			#img = img.resize([32,32],Image.ANTIALIAS)
			result_x = np.append(result_x,img,axis=0)
			print(result_x.shape)
			result_y.append(label)
			print(len(result_y))

	f.close()

	return result_x, result_y
