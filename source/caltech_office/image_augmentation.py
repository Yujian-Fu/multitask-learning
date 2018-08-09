import numpy as np
from PIL import Image, ImageEnhance, ImageOps, ImageFile
import random
import threading, os, time
import logging
from keras.utils import to_categorical
import config

from read_file import read_image

def randomCrop(image):
	image_width = image.shape[0]
	image_height = image.shape[1]
	crop_width = np.random.randint(0, 7)
	crop_height = np.random.randint(0, 7)
	random_region = (
    	crop_width, crop_height, 
    	crop_width+32, crop_height+32)
	print(random_region)
	image_PIL = Image.fromarray(image)
	image_PIL = image_PIL.crop(random_region)
	image = np.array(image_PIL)
	return image 

def augmentation(image, label):
	output_label = []
	for i in range(image.shape[0]//40):
		batch = image[40*i:40*(i+1),:]
		for j in range(10):
			if (i==0 and j==0):
				Cropresult = randomCrop(batch)
				output_image = Cropresult
				output_label.append(label[i])
				print("/////////")
				print(output_image.shape[0]/32,len(output_label))
			else:
				Cropresult = randomCrop(batch)
				output_image = np.append(output_image, Cropresult, axis=0)
				output_label.append(label[i])
				print(image.shape[0]/40,len(label))
				print(output_image.shape[0]/32,len(output_label))


	return output_image, output_label

def get_image(path):

	name = str(path.split('/')[-2])

	image, label = read_image(path)


	image, label = augmentation(image, label)
	print("type of image is:")
	print(type(image))
	print("shape of image is")
	print(image.shape)

	print("type of label is:")
	print(type(label))
	print("shape of label is")
	print(len(label))

	image = np.reshape(image, [-1,32,32])
	label = to_categorical(label, 66)

	np.save(name+"_data.npy",image)
	np.save(name+"_label.npy",label)




	#return np.expand_dims(image,axis=3), label


#'''
def main():
	get_image(config.train_path1)
	#get_image(config.train_path2)
	get_image(config.train_path3)
	get_image(config.train_path4)
	#get_image(config.test_path1)
	#get_image(config.test_path2)
	#get_image(config.test_path3)
	#get_image(config.test_path4)

if __name__ == "__main__":
	main()
#'''











