#!/usr/bin/env python
# -*- coding=utf-8 -*-
from threading import Thread
from Outline import contour
import json
import numpy as np
import os
from time import sleep
import rospy
import cv2
from sensor_msgs.msg import Image
os.chdir('/home/momo/catkin_ws/src/jurvis/scripts/Program/')
#
from Calibration.calibrate import Communicate_with_SCM, Sense_Self, ros_spinOnce

#
myCS = Communicate_with_SCM()
mySS = Sense_Self()
node_name = 'calibrate_node'
rospy.init_node(node_name)
image_sub = rospy.Subscriber("/camera/color/image_raw", Image, mySS.convert_RGB, buff_size=2097152)  # 2MB
depth_sub = rospy.Subscriber("/camera/depth/image_rect_raw", Image, mySS.convert_Depth, buff_size=2097152)
rospy.loginfo("Waiting for image topics...")

def main1(COOR):
    sleep(1)
    while True:
        ros_spinOnce()

        if mySS.debug_img:
            cv2.imshow('debug', mySS.debug_img)
        cv2.imshow('depth_rgb', mySS.frame_depth_rgb)
        cv2.imshow('rgb', mySS.frame_rgb)
        # cv2.imshow('depth_gray', mySS.frame_depth_gray)
        flag = cv2.waitKey(30) & 0xFF
        if COOR[0]=='record':
            xy = mySS.coor[:2]
            # print(xy)
            if None not in xy:
                COOR.append(xy)
        elif flag == 27 or 'end' == COOR[0]:
            rospy.signal_shutdown("User hit ESC key to quit.")
            cv2.destroyAllWindows()
            break

def main2(COOR):
    print('main2 start')
    myCS.home_arm()
    sleep(0.7)
    COOR[0] = 'record'
    for i in range(1000, 1400, 120):
        myCS.writeSCM([1200, 600, 1800, 1100, 1850, i], 700)
        if (myCS.readStatu()):
            pass
        pass
    COOR[0]=''
    myCS.home_arm()

    L=len(COOR)
    if L>=9:
        coor = [COOR[1], COOR[L // 2], COOR[-1]]
        # centers = []
        points = list(map(contour.Point, coor))
        # for i in range(len(coor) - 2):
        #     centers.append(contour.calculate_circle_center(*points[i:i + 3]))
        # print(centers)
        # final_cen = np.mean(centers, axis=0)
        # print(final_cen)
        final_cen=contour.Whereismy_center.calculate_circle_center(*points)
        final_cen=list(map(int,final_cen))
        print(final_cen, 'saved in', os.path.abspath('./'))
        with  open('arm_center.json','w') as fp:
            json.dump(final_cen, fp)
    else:
        print('数据不足，重新设置位置')
    print 'end'
    myCS.closeSerial()
    COOR[0]='end'
if __name__ == '__main__':
    #
    COOR = [None]#第0位当作通信口
    thread1 = Thread(target=main2, args=(COOR,))
    thread1.daemon = True
    thread1.start()
    try:
        main1(COOR)
    except :
        myCS.home_arm()
        myCS.closeSerial()
        exit()
