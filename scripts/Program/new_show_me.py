#!/usr/bin/env python
# -*- coding=utf-8 -*-

# 请合并到main.py
# 测试程序
# 可以观看计算机画出自己中心坐标
from threading import Thread
from Outline import contour
import json
import numpy as np
import os
from time import sleep
import rospy
import cv2
from sensor_msgs.msg import Image
from Detection import detect
os.chdir('/home/momo/Project/jurkis_ws/src/jurvis/scripts/Program/')
#
from Calibration.calibrate import Communicate_with_SCM, Sense_Self, ros_spinOnce,ArmEye_collectingData

#
myCS = Communicate_with_SCM()
mySS = Sense_Self()
myAE=ArmEye_collectingData()
myWaI=contour.Where_am_I()
node_name = 'calibrate_node'
rospy.init_node(node_name)
image_sub = rospy.Subscriber("/camera/color/image_raw", Image, mySS.convert_RGB, buff_size=2097152)  # 2MB
depth_sub = rospy.Subscriber("/camera/depth/image_rect_raw", Image, mySS.convert_Depth, buff_size=2097152)
rospy.loginfo("Waiting for image topics...")
sometime=0.7
# print('111')
def main1():
    sleep(sometime*2)
    mySS.waitRosMsg()
    print('main1')
    # cv2.namedWindow('chendushow',cv2.WINDOW_KEEPRATIO)
    k=0
    while True:
        ros_spinOnce()
        # cv2.imshow('depth_rgb', mySS.frame_depth_rgb)
        cv2.imshow('depth_gray', mySS.frame_depth_gray)
        # mySS.frame_rgb, _,_ = detect.minor(mySS.frame_rgb)
        # print("gg5")
        # mySS.frame_rgb=myWaI.draw_grasp_point_and_arm_center(mySS.frame_rgb,mySS.coor[:2])
        # print("gg6")
        # cv2.imshow('chendushow', mySS.frame_rgb)
        flag = cv2.waitKey(30) & 0xFF
        if flag == 27:
            rospy.signal_shutdown("User hit ESC key to quit.")
            cv2.destroyAllWindows()
            myCS.home_arm()
            break
        elif flag==32:
            print("save a jpg")
            cv2.imwrite(str(k)+".jpg",mySS.frame_depth_gray)
            k+=1

def main2():
    print('main2 start')
    myCS.home_arm()
    sleep(sometime)
    for p in myAE.para:
        myCS.writeSCM(p)
        if (myCS.readStatu()):
            sleep(0.3)
            pass
        pass
    sleep(1)
    print('end')
    myCS.home_arm()

if __name__ == '__main__':
    #
    detect.properity(480, 640)
    # thread1 = Thread(target=main2, args=())
    # thread1.daemon = True
    # thread1.start()
    try:
        main1()
    except :
        myCS.home_arm()
        myCS.closeSerial()
        exit()
