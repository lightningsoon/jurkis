#!/usr/bin/env python
# -*- coding=utf-8 -*-

import cv2
from calibrate import Sense_Self,ros_spinOnce
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from time import sleep
mySS=Sense_Self()
node_name = 'calibrate_node'
rospy.init_node(node_name)
image_sub = rospy.Subscriber("/camera/color/image_raw", Image, mySS.convert_RGB, buff_size=2097152)  # 2MB
depth_sub = rospy.Subscriber("/camera/depth/image_rect_raw", Image, mySS.convert_Depth, buff_size=2097152)
rospy.loginfo("Waiting for image topics...")
sleep(2)
mySS.waitRosMsg()
while True:
    ros_spinOnce()
    if mySS.debug_img:
        cv2.imshow('debug',mySS.debug_img)
    # cv2.imshow('depth_gray', mySS.frame_depth_gray)
    cv2.imshow('depth_rgb', mySS.frame_depth_rgb)
    cv2.imshow('rgb', mySS.frame_rgb)
    flag = cv2.waitKey(30) & 0xFF
    if flag == 27:
        rospy.signal_shutdown("User hit ESC key to quit.")
        cv2.destroyAllWindows()
        break

# roslaunch openni2_launch openni2.launch