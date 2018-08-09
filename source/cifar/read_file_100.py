# -*- coding:utf-8 -*-
import pickle as p
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as plimg
from PIL import Image
def load_CIFAR_batch(filename):
    """ load single batch of cifar """
    with open(filename, 'rb')as f:
        datadict = p.load(f,encoding='latin1')
        X = datadict['data']
        #batch_label = datadict['batch_label']
        fine_labels = datadict['fine_labels']
        coarse_labels = datadict['coarse_labels']

        X = X.reshape(-1, 32, 32, 3)
        #batch_label = np.array(batch_label)
        fine_labels = np.array(fine_labels)
        course_labels = np.array(coarse_labels)
    return  X, fine_labels, coarse_labels


def load_CIFAR_Labels(filename):
    with open(filename, 'rb') as f:
        lines = [x for x in f.readlines()]
        print(lines)


'''if __name__ == "__main__":
    #load_CIFAR_Labels("./cifar-10-batches-py/batches.meta")
    #imgX, imgY = load_CIFAR_batch("./cifar-10-batches-py/data_batch_1")
    img, batchlabel, label20, label100 = load_CIFAR_batch("./cifar-100-python/train")
    print(img.shape)
    print((batchlabel))
    print(len(label20))
    print(len(label100))

    print("正在保存图片:")
    for i in range(imgX.shape[0]):
        imgs = imgX[i - 1]
        if i < 100:#只循环100张图片,这句注释掉可以便利出所有的图片,图片较多,可能要一定的时间
            img0 = imgs[0]
            img1 = imgs[1]
            img2 = imgs[2]
            i0 = Image.fromarray(img0)
            i1 = Image.fromarray(img1)
            i2 = Image.fromarray(img2)
            img = Image.merge("RGB",(i0,i1,i2))
            name = "img" + str(i)
            img.save("./images/"+name,"png")#文件夹下是RGB融合后的图像
            for j in range(imgs.shape[0]):
                img = imgs[j - 1]
                name = "img" + str(i) + str(j) + ".png"
                print("正在保存图片" + name)
                plimg.imsave("./image/" + name, img)#文件夹下是RGB分离的图像

    print("保存完毕.")
    '''