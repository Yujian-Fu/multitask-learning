# -*- coding: utf-8 -*- 

import numpy as np 
from PIL import Image
import math
import config
from keras.utils import to_categorical

def read_image(path, num):
	f = open(path)
	line = f.readline()
	label = line.split(" ")[1].split("\n")[0]

	line = line.split(" ")[0]

	result_x = Image.open(line)
	result_y_1 = []
	result_y_2 = []
	result_y_1.append(int(label))
	result_y_2.append(int(num))

	while line!= '':
		line = f.readline()

		if line.split('\n') != ['']:
			label = line.split(" ")[1].split("\n")[0]

			line = line.split(" ")[0]
			img = Image.open(line)
			result_x = np.append(result_x, img, axis=0)
			result_y_1.append(int(label))
			result_y_2.append(int(num))

	result_x = result_x.reshape(-1,40,40,3)
	f.close()
	return result_x, result_y_1, result_y_2


def randomcrop(image):
	image_width = image.shape[1]
	image_height = image.shape[2]
	crop_width = np.random.randint(0, 7)
	crop_height = np.random.randint(0, 7)
	result = image[crop_width:crop_width+32, crop_height:crop_height+32, 0:3]
	return result

def augmentation(image, label1, label2, epoch):
	out_label1 = []
	out_label2 = []
	for i in range(image.shape[0]):
		batch = image[i,:,:,:]
		for j in range(epoch):
			if (i==0 and j==0):
				result = randomcrop(batch)
				out_image = result
				out_label1.append(label1[i])
				out_label2.append(label2[i])
			else:
				result = randomcrop(batch)
				out_image = np.append(out_image, result, axis=0)
				out_label1.append(label1[i])
				out_label2.append(label2[i])

	out_image = out_image.reshape(-1, 32, 32, 3)
	return out_image, out_label1, out_label2



if __name__ == "__main__":
	train1 = config.train_path1
	train2 = config.train_path2
	train3 = config.train_path3
	train4 = config.train_path4
	test1 = config.test_path1
	test2 = config.test_path2
	test3 = config.test_path3
	test4 = config.test_path4

	X1, Y11, Y12 = read_image(train1, 0)
	print(X1.shape)
	X2, Y21, Y22 = read_image(train2, 1)
	print(X2.shape)
	X3, Y31, Y32 = read_image(train3, 2)
	print(X3.shape)
	X4, Y41, Y42 = read_image(train4, 3)
	print(X4.shape)
	X = np.concatenate((X1, X2, X3, X4), axis=0)
	Y1 = np.concatenate((Y11, Y21, Y31, Y41), axis=0)
	Y2 = np.concatenate((Y12, Y22, Y32, Y42), axis=0)
	print(X.shape, Y1.shape, Y2.shape)
	X, Y1, Y2 = augmentation(X, Y1, Y2, 10)
	Y1 = to_categorical(Y1, 66)
	Y2 = to_categorical(Y2, 4)

	x1, y11, y12 = read_image(test1, 0)
	x2, y21, y22 = read_image(test2, 1)
	x3, y31, y32 = read_image(test3, 2)
	x4, y41, y42 = read_image(test4, 3)
	x = np.concatenate((x1, x2, x3, x4), axis=0)
	y1 = np.concatenate((y11, y21, y31, y41), axis=0)
	y2 = np.concatenate((y12, y22, y32, y42), axis=0)
	x, y1, y2 = augmentation(x, y1, y2, 1)
	y1 = to_categorical(y1, 66)
	y2 = to_categorical(y2, 4)

	print("the shape of data is:")
	print(X.shape, Y1.shape, Y2.shape)
	print(x.shape, y1.shape, y2.shape)

	np.save("train_data", X)
	np.save("train_label1", Y1)
	np.save("train_label2", Y2)
	np.save("test_data", x)
	np.save("test_label1", y1)
	np.save("test_label2", y2)







