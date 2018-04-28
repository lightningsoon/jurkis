# -*- coding:utf-8 -*-
import cv2
import math
import numpy as np

width,height,channel=640,480,3

# 245,160 385,305

# img_deep=cv2.imread('/Users/huanghao/PycharmProjects/Jurvis/Program/Outline/data/deep0.png')
# img_bgr=cv2.imread('/Users/huanghao/PycharmProjects/Jurvis/Program/Outline/data/rgb0.png')
#
# cv2.imwrite('./deep.jpg',img_deep[160-5:305+10,245:385+20])
# cv2.imwrite('./bgr.jpg',img_bgr[160:305,245:385])

# 二值掩膜
size = 3
kernel = np.ones((size, size), dtype=np.uint8)


def binaryMask(image_rgb):
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 2)  # 高斯滤波
    # blur = cv2.bilateralFilter(gray, 9, 75, 75)  # 双边滤波
    res = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 0)
    res = cv2.morphologyEx(res, cv2.MORPH_CLOSE, kernel)
    return res


def circle(img_bgr, img_tgt,circle_point):
    # 预处理
    # 提取轮廓，得到唯一边缘点
    # 边缘提取
    square = img_bgr.shape[0] * img_bgr.shape[1]
    img_bin = binaryMask(img_bgr)
    _, contours, hierarchy = cv2.findContours(img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if len(cnt) > 50:
            try:
                ellipse = cv2.fitEllipse(cnt)
            except cv2.error:
                continue
            S1 = cv2.contourArea(cnt)
            S2 = math.pi * ellipse[1][0] * ellipse[1][1] / 4
            if 0.93 < S1 / S2 < 1.07 and 0.9 > S2 / square > 0.3:
                cv2.ellipse(img_tgt, ellipse, (255, 0, 0), 2)
                return img_tgt,ellipse
    # cv2.imshow('tmp', img_bin)
    return img_tgt,None
    pass

def which_kind_is(img):
    #@小弘

    pass
class Point(object):
    def __init__(self,coor_in_img=None):
        '''

        :param coor_in_img: 按照图像的坐标系来
        '''
        self.x=coor_in_img[0] if coor_in_img else None
        self.y=coor_in_img[1] if coor_in_img else None
    pass
class Where_am_I(object):
    def __init__(self):
        self.filename = 'arm_center.json'
        self.arm_center=self.read_self_arm_center()
        self.generate_new_background()
    def read_self_arm_center(self):

        import json
        import os
        if os.path.isfile(self.filename):
            with open(self.filename,'r') as f:
                arm_center=json.load(f)
                # print(arm_center)
                return arm_center
        else:
            print("%s文件不存在" % (self.filename))
            print(os.path.abspath('.'))
            exit()
    def save_arm_center(self,**data):
        '''

        :param data: dict={'x':int,'y':int}
        :return:
        '''
        import json
        with open(self.filename, 'w') as f:
            json.dump({'x':data['x'],'y':data['y']},f)
    def calculate_circle_center(self,pt1,pt2,pt3):
        '''
        目前只支持传入三个点求圆心
        :param pt1: 三个点
        :param pt2:
        :param pt3:
        :return: 圆心 x,y
        '''
        import math
        cp = Point()
        A1 = pt1.x - pt2.x
        B1 = pt1.y - pt2.y
        C1 = (pow(pt1.x, 2) - pow(pt2.x, 2) + pow(pt1.y, 2) - pow(pt2.y, 2)) / 2
        A2 = pt3.x - pt2.x
        B2 = pt3.y - pt2.y
        C2 = (pow(pt3.x, 2) - pow(pt2.x, 2) + pow(pt3.y, 2) - pow(pt2.y, 2)) / 2
        # 为了方便编写程序，令temp = A1 * B2 - A2 * B1
        temp = A1 * B2 - A2 * B1
        # 判断三点是否共线
        if (temp == 0):
        # 共线则将第一个点pt1作为圆心
            cp.x = pt1.x
            cp.y = pt1.y

        else :
        # 不共线则求出圆心：
        # center.x = (C1 * B2 - C2 * B1) / A1 * B2 - A2 * B1
        # center.y = (A1 * C2 - A2 * C1) / A1 * B2 - A2 * B1
            cp.x = (C1 * B2 - C2 * B1) / temp
            cp.y = (A1 * C2 - A2 * C1) / temp
        # radius = math.sqrt((cp.x - pt1.x) * (cp.x - pt1.x) + (cp.y - pt1.y) * (cp.y - pt1.y))
        cp.x,cp.y=int(cp.x),int(cp.y)

        return (cp.x,cp.y)

    def calculate_grasp_point(self,point_set):
        '''
        计算抓取点
        也就是离机械臂圆心最近点
        不信画个图，图形学
        :param point_set:
        :return:
        '''
        point_set=np.array(point_set)
        arm_center=np.array(self.arm_center)
        distances=np.linalg.norm(point_set-arm_center,axis=1)
        # print(distances)
        mindis_index=np.argmin(distances)
        return (point_set[mindis_index],distances[mindis_index])
    #图像大小
    def generate_new_background(self):
        import numpy as np
        new_width,new_height=width,height
        arm_center=self.arm_center
        motion_x,motion_y=0,0
        if arm_center[0]>width:
            new_width=arm_center[0]
        elif arm_center[0]<0:
            motion_x=0-arm_center[0]
            arm_center[0]=0
            new_width=width+motion_x
        if arm_center[1]>height:
            new_height=arm_center[1]
        elif arm_center[1]<0:
            motion_y=0-arm_center[1]
            arm_center[1]=0
            new_height=height+motion_y
        self.motion_x, self.motion_y = motion_x, motion_y
        self.frame_with_arm_and_direction=np.zeros((new_height,new_width,channel),dtype=np.uint8)
        self.frame_with_arm_and_direction = cv2.drawMarker(self.frame_with_arm_and_direction, tuple(self.arm_center),
                                    (0, 255, 255), cv2.MARKER_DIAMOND, 15, 4)
    def draw_grasp_point_and_arm_center(self,frame,grasp_point):
        '''
        机械臂中心很可能在图像之外，所以画图时候得注意
        从红色指向绿色
        :param frame: bgr图像
        :param grasp_point: (x,y)
        :return:frame_with_arm_and_direction
        '''
        background=self.frame_with_arm_and_direction.copy()
        background[self.motion_y:self.motion_y + height, self.motion_x:self.motion_x + width, :]=frame

        if np.all(np.array(grasp_point) != None):
            background = cv2.drawMarker(background, tuple(grasp_point),
                                                          (0, 0, 255), cv2.MARKER_STAR, 12)
            grasp_point=(grasp_point[0]+self.motion_x,grasp_point[1]+self.motion_y)

            background=cv2.line(background,tuple(self.arm_center),tuple(grasp_point),
                                                  (255,0,0),2,cv2.LINE_AA)
        return background