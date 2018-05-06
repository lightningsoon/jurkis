# coding=utf-8
from contour import binaryMask,kernel,size
import contour
import os
import cv2
import math

# 此程序测试contour中找圆圈的函数是否好用，主要是收集数据，调整参数
workdir='/Users/huanghao/Desktop/file/rgb/'

def debug(img):
    cv2.imshow('',img)
    cv2.waitKey(0)
imgfiles=os.listdir(workdir)[:1]
imgs=list(cv2.imread(workdir+im) for im in imgfiles)
for im in imgs:
    res=binaryMask(im)
    # debug(res)
    _, conts, hierarchy = cv2.findContours(res, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    square = res.shape[0] * res.shape[1]#图像面积
    print("图像面积（像素）：",square)
    leng_dict={}
    for cnt in conts:
        try:
            leng_dict[len(cnt)]+=1
        except KeyError:
            leng_dict[len(cnt)]=1
        if len(cnt) >= 50:#一个椭圆至少需要5个点
            try:
                ellipse = cv2.fitEllipse(cnt)
            except :
                continue
            S1 = cv2.contourArea(cnt)# 点围起来圈点面积
            S2 = math.pi * ellipse[1][0] * ellipse[1][1] / 4#椭圆面积
            # 下面这个根据大量实验得出了参数，第二个条件适合碗的口径
            try:
                if 0.93 < S1 / S2 < 1.07 and 0.9 > S2 / square > 0.25:

                    im1= cv2.ellipse(im, ellipse, color=(130, 80, 0), thickness=2)
                    cv2.imshow('1',im1)
                    cv2.waitKey(0)
                    exit()
            except ZeroDivisionError:
                print("报错",S1,S2)
            im = cv2.drawContours(im,[cnt],-1,(0,222,0))
            cv2.imshow('',im)
            cv2.waitKey(0)
            print(len(cnt),S1, S2 )
    print(leng_dict)
    # cv2.imshow('',im)
    # cv2.waitKey(0)