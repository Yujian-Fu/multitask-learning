# -*- coding: utf-8 -*- 
import os 
import cv2

#def Test1(rootDir): 
list_dirs = os.walk("/home/fyj/桌面/Multitask_learning/dataset/") 
for root, dirs, files in list_dirs: 
    #for d in dirs: 
        #print os.path.join(root, d)      
    for f in files: 
        #print os.path.join(root, f)
        filename = os.path.join(root, f)
        image = cv2.imread(filename) 
        if (filename.split('.')[-1]=="jpg"):
        	res = cv2.resize(image, (40,40),interpolation=cv2.INTER_CUBIC)
        	cv2.imwrite(filename, res)

