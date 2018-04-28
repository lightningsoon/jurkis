#!/usr/bin/env python
# -*- coding=utf-8 -*-
# 爱咖啡开始

import rospy
import sys
import cv2

#import cv2.cv as cv
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
cv=cv2
class cvBridgeDemo():
    def __init__(self):
        self.node_name = "cv_bridge_demo"
        rospy.init_node(self.node_name)

        rospy.on_shutdown(self.cleanup)

        self.cv_window_name = self.node_name

        self.bridge = CvBridge()

        self.image_sub = rospy.Subscriber("/camera/rgb/image_raw", Image, self.image_callback)
        self.depth_sub = rospy.Subscriber("/camera/depth/image_raw", Image, self.depth_callback)
        
        rospy.loginfo("Waiting for image topics...")

        self.i=0
        self.frame=None
        self.depth_frame=None
        print('???')
        while (np.all(self.frame==None) or np.all(self.depth_frame==None)):
            print(self.frame)
            rospy.rostime.wallsleep(0.1)
            pass
        while True and not rospy.is_shutdown():
            # print(1)
            rospy.rostime.wallsleep(0.03)
            # print('!!!')
            # print(self.frame)
            # print(self.depth_frame)

            cv2.imshow('RGB', self.frame)
            cv2.imshow("Depth Image", self.depth_frame)
            self.keystroke = cv.waitKey(30)
            if 32 <= self.keystroke and self.keystroke < 128:
                cc = chr(self.keystroke).lower()
                if cc == 'q':
                    # The user has press the q key, so exit
                    rospy.signal_shutdown("User hit q key to quit.")
                if self.keystroke == 32:
                    cv2.imwrite('deep' + str(self.i) + '.png', self.depth_frame)
                    cv2.imwrite('rgb' + str(self.i) + '.png', self.frame)
                    self.i += 1
    def image_callback(self, ros_image):
        # global frame,depth_array,display_image

        # Use cv_bridge() to convert the ROS image to OpenCV format
        try:
            frame = self.bridge.imgmsg_to_cv2(ros_image, "bgr8")
        except CvBridgeError, e:
            print e
        
        # Convert the image to a Numpy array since most cv2 functions
        # require Numpy arrays.
        self.frame = np.array(frame, dtype=np.uint8)
        # cv2.imshow('RGB',frame)
        # cv2.imshow("Depth Image", depth_array)
        #
        # self.keystroke = cv.waitKey(30)
        # if 32 <= self.keystroke and self.keystroke < 128:
        #     cc = chr(self.keystroke).lower()
        #     if cc == 'q':
        #         # The user has press the q key, so exit
        #         rospy.signal_shutdown("User hit q key to quit.")
        #     if self.keystroke == 32:
        #
        #         cv2.imwrite('deep'+str(self.i)+'.png',depth_array)
        #         cv2.imwrite('rgb'+str(self.i)+'.png',frame)
        #         self.i+=1
        #
    def depth_callback(self, ros_image):
        # global frame,depth_array,display_image

        # Use cv_bridge() to convert the ROS image to OpenCV format
        try:
            # The depth image is a single-channel float32 image
            depth_image = self.bridge.imgmsg_to_cv2(ros_image, "32FC1")
            # print(depth_image)
            # print()
        except CvBridgeError, e:
            print e
        # Convert the depth image to a Numpy array since most cv2 functions
        # require Numpy arrays.
        self.depth_frame = np.array(depth_image, dtype=np.uint8)

        # Normalize the depth image to fall between 0 (black) and 1 (white)
        # cv2.normalize(depth_array, depth_array, 0, 1, cv2.NORM_MINMAX)

    
    def cleanup(self):
        print "Shutting down vision node."
        cv2.destroyAllWindows()   
    
def main(args):       
    try:
        cvBridgeDemo()
        rospy.spin()
    except KeyboardInterrupt:
        print "Shutting down vision node."
        cv.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)

