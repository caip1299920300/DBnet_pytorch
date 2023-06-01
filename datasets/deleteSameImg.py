# -*- coding:utf-8 -*-
'''
@Author: siyuan
@功能: 无
@输入参数: 无
@输出参数: 无
@注意要点: 无
'''

import shutil
import numpy as np
from PIL import Image
import os
import cv2


# 输入灰度图，返回hash
def getHash(img, imgSize=(8,8)):
    img = cv2.resize(img, imgSize)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    avreage = np.mean(img)  #计算像素平均值
    hash = []
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i, j] > avreage:
                hash.append(1)
            else:
                hash.append(0)
    return hash


# 计算汉明距离
def Hamming_distance(hash1, hash2, size=8):
    num = 0
    for index in range(len(hash1)):
        if hash1[index] == hash2[index]:
            num += 1
    score = num/(size*size)
    return score


def compareSameByHash(srcImg, testImg, size=8, channle=0):
    img1 = cv2.resize(srcImg, (size, size))
    img2 = cv2.resize(testImg, (size, size))
    channle = int(min(3, max(0, channle)))
    hsv_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
    hsv_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)

    imgGrey1 = hsv_img1[:, :, channle].copy()
    imgGrey2 = hsv_img2[:, :, channle].copy()
    # 获取哈希
    hash1 = getHash(imgGrey1)
    hash2 = getHash(imgGrey2)

    score= Hamming_distance(hash1, hash2, size)
    return score

def comperByte(dir_image1, dir_image2):
    with open(dir_image1, "rb") as f1:
        size1 = len(f1.read())
    with open(dir_image2, "rb") as f2:
        size2 = len(f2.read())
    if (size1 == size2):
        result = "大小相同"
    else:
        result = "大小不同"
    return result


def comperSize(dir_image1, dir_image2):
    image1 = Image.open(dir_image1)
    image2 = Image.open(dir_image2)
    if (image1.size == image2.size):
        result = "尺寸相同"
    else:
        result = "尺寸不同"
    return result


imgsData = []#{"imgName":"xxx.jpg","shape": (10,10,3), "hashList":[8*8]}
if __name__ == '__main__':

    load_path = r'./new'  # 要去重的文件夹
    save_path = r'./new_qc'  # 空文件夹，用于存储检测到的重复的照片
    os.makedirs(save_path, exist_ok=True)

    # 获取图片列表 file_map，字典{文件路径filename : 文件大小image_size}
    file_map = {}
    image_size = 0
    # 遍历filePath下的文件、文件夹（包括子目录）
    for parent, dirnames, filenames in os.walk(load_path):
        for filename in filenames:
            path = os.path.join(parent, filename)
            img = cv2.imread(path)
            if img is None:
                continue

            hashList =  getHash(img)
            imgsData.append({"imgName":filename, "shape": img.shape, "hashList":hashList})

    # 取出重复的图片
    imgNum = len(imgsData)
    for i, data in enumerate(imgsData):
        size1 = data["shape"]
        for j in range(imgNum):
            if j == i:
                continue
            # if data["shape"] != imgsData[j]["shape"]:
            #     continue
            score = Hamming_distance(data["hashList"], imgsData[j]["hashList"])
            if score < 0.85:
                continue
            print("相同的图片： {0}   {1}".format(data["imgName"], imgsData[j]["imgName"]))
            img1 = cv2.imread(os.path.join(load_path, data["imgName"]))
            img2 = cv2.imread(os.path.join(load_path, imgsData[j]["imgName"]))
            if img1 is None or img2 is None:
                continue
            cv2.imshow("img1", img1)
            cv2.imshow("img2", img2)
            keyVal = cv2.waitKey(0)
            if keyVal & 0xff == ord('s'):#ord函数是取其ASCII编码
                if len(data["imgName"])>10:
                    shutil.move(os.path.join(load_path, imgsData[j]["imgName"]), os.path.join(save_path, imgsData[j]["imgName"]))
                    continue
                elif len(imgsData[j]["imgName"])>10:
                    shutil.move(os.path.join(load_path, data["imgName"]), os.path.join(save_path, data["imgName"])) 
                    continue
                    
                if (img1.shape[0] * img1.shape[1] < img2.shape[0] * img2.shape[1]):# 保留大的图片
                    shutil.move(os.path.join(load_path, data["imgName"]), os.path.join(save_path, data["imgName"])) 
                else:
                    shutil.move(os.path.join(load_path, imgsData[j]["imgName"]), os.path.join(save_path, imgsData[j]["imgName"]))

