#!/usr/bin/env python
# -*- coding=utf-8 -*-
from __future__ import division
import cv2
import numpy as np
import csv
import time
from threading import Thread
from Queue import Queue
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import logging
import sys
'''
自身功能：
    采集数据
        图像识别亮点
        读取下位机角度
被调用功能：
    目标检测
        颜色空间
        二值化
        形状匹配->要改
'''
logger=logging.getLogger()
logger.setLevel(logging.DEBUG)
format = logging.Formatter("%(asctime)s - %(message)s")  # output format
sh = logging.StreamHandler(stream=sys.stdout)  # output to standard output
sh.setFormatter(format)
logger.addHandler(sh)

class Communicate_with_SCM(object):
    def __init__(self):
        # 与单片机通信
        self.ser = self.openSerial()
        # self.openSerial()
        self.home_arm()
        logger.info(self.ser.readline())
        # logger.info(self.ser.readline())
        print ''

        pass

    def openSerial(self):
        import serial
        from serial.tools import list_ports
        plist = list(list_ports.comports())
        if len(plist) <= 0:
            logger.error('no port')
            exit()
        else:
            ser = serial.Serial(port=plist[0].device, baudrate=115200)
            # ser.flush()
            # ser.timeout=0.5
            # ser.close()
            # TODO 轮询port
            if not ser.isOpen():
                ser.open()
            # ser.flush()
            # ser.timeout=None
            logger.info('require serial opened')
            # ser.flush()
            time.sleep(2)# 等端口打开，官网说的!!
            ser.timeout=1
            self.ser=ser
            return ser
        pass

    def closeSerial(self):
        self.ser.close()
        logger.info('serial closed')

    def writeSCM(self, something=[1468, 600, 1100, 1000, 1350, 1060], time=1000):
        '''
        将列表转成bytes 加上 \n
        :param something: list
        :param time: int 动作运行时间，ms
        :return:
        '''
        if len(something) == 7:
            # （6个参数和一个时间）
            pass
        elif len(something) == 6:
            something = something + [time]
        something = list(map(str, something))
        something = ','.join(something)
        logger.info('send %s to SCP' % (something))
        self.ser.write(something + '!')

    def readStatu(self):
        time.sleep(2)
        for i in range(5):
            sign = self.ser.readline()
            if sign == '1\r\n':
                return True
            else:
                print('SCP reported ??? something : %s' % sign)
        logger.error('timeout with SCP in readStatu')
        self.closeSerial()
        exit()

    def home_arm(self):
        # 归位
        logger.info('homing')
        self.ser.write('homing' + '!')


class ArmEye_collectingData(object):
    '''
    记录数据
    训练模型
    画空间图
    '''

    def __init__(self,openfile=False):
        self.__point_number=3#点数量
        if openfile:
            self.openCSV()
        self.inform()
        self.para = self.generatePara()
        pass

    def inform(self):
        para = self.generatePara()
        self.__N=len(list(para))
        print("need time: %.2f min" % (self.__N* 2 / 60))#2s运动时间
        self.__count = 0

    def openCSV(self):
        self.__f_para = open('/home/momo/catkin_ws/src/jurvis/scripts/Program/Calibration/label.csv', 'a')  # 数据
        self.__f_coor = open('/home/momo/catkin_ws/src/jurvis/scripts/Program/Calibration/data.csv', 'a')  # 标签
        self.writer_para = csv.writer(self.__f_para)
        self.writer_coor = csv.writer(self.__f_coor)

    def saveCSV(self):
        self.__f_para.close()
        self.__f_coor.close()

    def write2CSV(self, parameter, coordination):
        # 保存数据到文件，并隔几次保存一下
        self.__count += 1
        print('finished %d / %d all \n %s,%s' % (self.__count, self.__N, str(parameter), str(coordination)))
        self.writer_para.writerow(parameter)
        self.writer_coor.writerow(coordination)
        if self.__count % 20 == 0:
            print('++++++++++++++++saved CSV')
            self.saveCSV()
            self.openCSV()

    def generatePara(self):
        # 电机运动范围
        left,right,num=1360,640,self.__point_number#a[5]
        lr=(left,right,(right-left)//num)
        a = [None] * 6
        # TODO 范围有待确定（未完成）
        a[0] = 1500# 爪子抓紧
        a[1] = 600# 爪子摆正
        R4=(1750,1950,(1950-1750)//num)# (1900,2170) a[4]
        R3=(1500,1200,-(1400-1100)//num) # (1000,1300) a[3]
        R2=(1250,1800,(1800-1250)//num)# (1200,1900) a[2]
        for a[5] in range(*lr):
            for a[2] in range(*R2):
                for a[3] in range(*R3):
                    for a[4] in range(*R4):
                        yield a


class Sense_Self(object):
    def __init__(self,flag4can_put_coor=False):
        from cv2 import aruco
        self.__bridge = CvBridge()
        # 
        self.flag4can_put_coor = False  # 数据存放开关
        # 图像类
        self.coor = [None] * 3  # 相机
        self.__x = [None] * 3 # 临时三维相机坐标
        self.__z = None#相机深度
        # marker
        self.__marker_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_50)
        # 显示
        self.frame_rgb = None
        self.__frame_depth_row = None
        self.__depth_array = np.ones((480, 640, 3), dtype=np.uint8)
        self.__depth_array[:, :, 1] = self.__depth_array[:, :, 1] * 255
        self.__depth_array[:, :, 2] = self.__depth_array[:, :, 2] * 180
        self.frame_depth_gray = None
        self.frame_depth_rgb = None
        self.debug_img=None
        pass

    def processIMG(self, frame):
        from cv2 import aruco
        '''
        获得图像中的标记，画出来
        存在x中
        若没有找到marker则x=[None,None,None]
        :param frame:
        :return:
        '''
        # TODO 二维码放正面，代码改成中点 （未完成）
        corners, ids, rejectedImgPoints = aruco.detectMarkers(frame, self.__marker_dict)
        # corner =[[左上，右上，右下，左下]]

        # print(np.any(circles!=None))
        if len(corners)==1:
            c_temp = np.reshape(corners[0], (1, -1, 1, 2)).astype(np.int32)# [1,4,1,2]
            cv2.polylines(frame, c_temp, True, (250, 0, 0), 2)
            self.__x[:2] = tuple((corners[0][0][0].astype(int)+corners[0][0][3].astype(int))//2)#1,2两点之间
            frame = cv2.circle(frame,tuple(self.__x[:2]), 4, (0, 0, 250), 2)#2表示顺序
            # print('__x',self.__x)
            # TODO 深度值
            self.__x[2] = self.getDeepth()
            self.coor = self.__x
            # print('if1',self.coor)
        else:
            self.coor = [None] * 3
        if self.flag4can_put_coor and not np.any(np.array(self.coor) == None):
            # print('似乎存放了数据')/
            if not q.full():
                q.put_nowait(self.coor)
            self.flag4can_put_coor = False
        # print('?????????????',frame)
        # self.debug_img
        return frame



    def getDeepth(self):
        return int(self.__frame_depth_row[self.__x[1],self.__x[0]])
        pass

    # TODO ROS订阅
    def convert_RGB(self, img):
        try:
            frame = self.__bridge.imgmsg_to_cv2(img, "bgr8")
        except CvBridgeError, e:
            print e
            exit()
        frame = np.array(frame, dtype=np.uint8)
        self.frame_rgb = self.processIMG(frame)

    # TODO ROS订阅
    def convert_Depth(self, img):
        # TODO ???
        try:
            # The depth image is a single-channel float32 image
            depth_image = self.__bridge.imgmsg_to_cv2(img, "32FC1")
        except CvBridgeError, e:
            print e
            exit()
        # print(depth_image)
        self.__frame_depth_row = depth_image
        # TODO 解算距离-ros_cv2__bridge
        self.__depth_array[:, :, 0] = np.array(depth_image * 180 / 10000, dtype=np.uint8)  # hsv
        self.frame_depth_rgb = cv2.cvtColor(self.__depth_array, cv2.COLOR_HSV2BGR)
        self.frame_depth_gray = np.array(depth_image * 255 / 10000, dtype=np.uint8)  # gray
        pass

    def waitRosMsg(self):
        from rospy.core import logdebug
        # import os
        if not rospy.core.is_initialized():
            raise rospy.exceptions.ROSInitException("client code must call rospy.init_node() first")
        # logdebug("node[%s, %s] entering spin(), pid[%s]", rospy.core.get_caller_id(), rospy.core.get_node_uri(),
        #          os.getpid())
        while (np.all(self.frame_rgb == None) or np.all(self.frame_depth_rgb == None) or np.all(self.frame_depth_gray == None)):
            print('rgb:',self.frame_rgb,'d-rgb:',self.frame_depth_rgb,'d-gray',self.frame_depth_gray)
            rospy.rostime.wallsleep(0.8)
            pass

def main():
    # 开启线程
    mythread = Thread(target=assist)
    mythread.daemon=True
    mythread.start()
    logger.info('thread started')
    mySense_Self.waitRosMsg()
    while True:
        ros_spinOnce()
        cv2.imshow('rgb', mySense_Self.frame_rgb)
        # cv2.imshow('depth_gray', mySense_Self.frame_depth_gray)
        cv2.imshow('depth_rgb', mySense_Self.frame_depth_rgb)
        flag = cv2.waitKey(30) & 0xFF
        if flag == 27:
            rospy.signal_shutdown("User hit ESC key to quit.")
            interrupt()
            break
        elif flag == 32:
            # 记录数据
            # mySense_Self.flag4can_put_coor = True
            pass


def assist():
    logger.info('----------Assist----------')
    for p in myArmEye_collectingData.para:
        myCommunicate_with_SCM.writeSCM(p)
        # 等待到达
        coor = None
        mySense_Self.flag4can_put_coor=True
        if (myCommunicate_with_SCM.readStatu()):
            time.sleep(0.1)
            # mySense_Self.flag4can_put_coor = True
            # 等待数据返回
            # print(p)
            if not q.empty():
                coor = q.get()
            # logger.debug(coor)
            if coor:
                myArmEye_collectingData.write2CSV(p, coor)
            pass
        pass
    print('!!!!!!!!!!!!!!!exhaust parameters')
    time.sleep(2)
    myCommunicate_with_SCM.home_arm()


def interrupt():
    cv2.destroyAllWindows()
    Communicate_with_SCM.closeSerial()
    myArmEye_collectingData.saveCSV()

def ros_spinOnce():
    #给外部函数用的
    if rospy.is_shutdown():
        exit()
    rospy.rostime.wallsleep(0.03)



# def performance_test(func):
#     t0=time.time()
#     def decorator():
#         @functools.wraps(func)
#         def wrapper(*args, **kw):
#             return func(*args, **kw)
#         return wrapper
#     print(func.__name__,':',time.time()-t0)
#     return decorator
q = Queue(1)
if __name__ == u'__main__':
    def ros_spinOnce():
        if rospy.is_shutdown():
            interrupt()
            exit()
        rospy.rostime.wallsleep(0.03)
    #
    mySense_Self = Sense_Self()
    myArmEye_collectingData = ArmEye_collectingData(openfile=True)
    myCommunicate_with_SCM = Communicate_with_SCM(True)
    #
    node_name = 'calibrate_node'
    rospy.init_node(node_name)
    rospy.on_shutdown(interrupt)
    image_sub = rospy.Subscriber("/camera/color/image_raw", Image, mySense_Self.convert_RGB, buff_size=2097152)  # 2MB
    depth_sub = rospy.Subscriber("/camera/depth/image_rect_raw", Image, mySense_Self.convert_Depth, buff_size=2097152)
    rospy.loginfo("Waiting for image topics...")
    try:
        main()
        # rospy.spin()
    except KeyboardInterrupt:
        interrupt()
    pass

'''
    t=geneCoor()
    for c in t:
        print(c)
        time.sleep(0.1)
'''
