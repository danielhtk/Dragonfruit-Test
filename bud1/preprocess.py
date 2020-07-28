import numpy as np
import cv2
import os
from matplotlib import pyplot as plt

base_dir="images/"

#創
def ImgPreprocess(path=base_dir):
    if not os.path.exists(path):
        os.makedirs(path)
        os.makedirs(os.path.join(path,"original"))
        os.makedirs(os.path.join(path,"generated"))
        os.makedirs(os.path.join(path,"result_pics"))
        print("Please put the pictures in {0}/original".format(path))
    else:
        directory="images/original/"
        for filename in os.listdir(directory):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                in_filename = os.path.join(directory, filename)
                img = cv2.imread(in_filename)
                ##須針對不同的資料來源進行調整，未來可用yolo補齊
                # img = img[300:850,620:1170] #照片
                #img = img[330:860,700:1050] #照片
                # img=img[500:1400,1080:1500]#影片截圖
                # res = cv2.resize(img,(512, 512), interpolation = cv2.INTER_CUBIC)
                # cv2.imwrite(os.path.join(path,"generated/") + os.path.splitext(os.path.basename(in_filename))[0] + ".png",img)
                ##change format
                cv2.imwrite(os.path.join(path,"generated/") + os.path.splitext(os.path.basename(in_filename))[0] + ".png",img)

def ImgPreprocess_1(path="images/generated"):
    if not os.path.exists(path):
        print("Please generate pictures in {0}/generated".format(path))
    else:
        directory = "images/generated/"
        for filename in os.listdir(directory):
            if filename.endswith(".png") or filename.endswith(".jpg"):
                in_filename = os.path.join(directory, filename)
                img = cv2.imread(in_filename)
                ##須針對不同的資料來源進行調整，未來可用yolo補齊
                # img = img[300:850,620:1170] #照片
                # img = img[330:860,700:1050] #照片
                # img=img[500:1400,1080:1500]#影片截圖
                # res = cv2.resize(img,(512, 512), interpolation = cv2.INTER_CUBIC)
                # cv2.imwrite(os.path.join(path,"generated/") + os.path.splitext(os.path.basename(in_filename))[0] + ".png",img)
                ##change format
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                #ret, thresh1 = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
                thresh1 = cv2.bilateralFilter(gray, 8,50,50)
                thresh1 = cv2.adaptiveThreshold(thresh1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 7, 7)
                #thresh1 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
                cv2.imwrite(os.path.join(base_dir, "generated/") + os.path.splitext(os.path.basename(in_filename))[0] + ".png", thresh1)

if __name__ == '__main__':
    ImgPreprocess()
    ImgPreprocess_1()
