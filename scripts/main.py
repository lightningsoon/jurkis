#!/usr/bin/env python
# -*- coding=utf-8 -*-

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
from Program.Detection.detect import minor, properity
from Program.Calibration.calibrate import Sense_Self, ros_spinOnce, Communicate_with_SCM
from Program.Calibration.EyeArm_Learning import Pinocchio
import Queue
from time import sleep
from threading import Thread
from Program.Outline import contour,cluster
cluster.restore_model()
fourcc = cv2.VideoWriter_fourcc(*'XVID')


# out = cv2.VideoWriter('output.mp4', fourcc, 30.0, (640, 480))


class Worker(Sense_Self, Pinocchio):
    def __init__(self):
        super(Worker, self).__init__()
        # TODO 应该要给True
        self.wait_or_act = False  # True 等待参数，False正在行动
        self.grasp_point=[None]*3
        self.target = None
        self.kind = None  # 种类名字
        self.num = [0]*5  # 每种数量
        # TODO 填写可以摆放的位置
        self.__setPosi = [1200,1000,800,600,400]
        pass

    def convert_Color(self, img):
        try:
            img = cb.imgmsg_to_cv2(img, 'bgr8')
        except CvBridgeError, e:
            print e
        self.frame_rgb = np.array(img, dtype=np.uint8)

    def working(self):
        '''
        5 工作部分，接在图像转格式存储后面
        :return:
        '''
        result, self.grasp_point[:2],self.kind = minor(self.frame_rgb)# 把数据带到外面来
        if self.grasp_point[:2]!=[None,None] and myWorker.wait_or_act==True:#可以执行并且，机械臂等待中
            self.grasp_point[2]=self.getDeepth()
            myWorker.wait_or_act=False#开始让子线程工作
        return result
        pass

    def execute_a_set_of_action(self, target):
        '''
        这次就手动设置

        :param target:[2:6]
        :return:
        '''
        cave=1400# TODO 坑的位置
        # 统计一下数量情况
        if self.num[self.kind]<5:
            elevate=1950-self.num[self.kind]*100
            self.num[self.kind]+=1
            # TODO 以后依靠识别，配合有监督示范
            # TODO 以后在初始化时，写到下位机
            actions0 = []
        else:
            #到了5只碗高，推一下
            elevate=1950
            actions0=[
                [1200, 1600, 1300, 1000, elevate, self.__setPosi[self.kind], 400],  # 转回
                [1200, 1600, 1700, 1200, elevate, self.__setPosi[self.kind], 400],  # 推出
                [1200, 1600, 1200, 1000, 1350, self.__setPosi[self.kind], 300]  # 归位
            ]
        actions=actions0+\
                [
                    [1200, 600] + target + [600],  # 到达
                    [1700, 600] + target + [50],  # 合上爪子
                    [1700, 600, 1900, 1000, 1500, target[3], 500],  # 抬高
                    [1700, 600, 1900, 1000, 1500, cave, 700],  # 到桶上
                    [1700, 2500, 1900, 1000, 1500, cave, 500],  # 倾倒
                    # [1700, 2500, 1800, 1000, 1500, cave, 20],  # 倾倒-抖一下
                    # [1700, 2500, 2000, 1000, 1500, cave, 20],  # 倾倒-抖一下
                    [1700, 600, 1900, 1000, 1500, self.__setPosi[self.kind], 400],  # 转回
                    [1700, 600, 1900, 1000, elevate, self.__setPosi[self.kind], 600],  # 下落
                    [1300, 600, 1900, 1000, elevate, self.__setPosi[self.kind], 800],  # 摆放
                    [1200, 600, 1200, 1000, 1350, self.__setPosi[self.kind], 600]  # 归位
                ]
        return actions
        pass



    pass


def clean():
    cv2.destroyAllWindows()
    print 'shut down'


def main():
    print('start')
    while True:
        # 4 获取图像
        ros_spinOnce()
        # 5 处理数据
        res = myWorker.working()
        cv2.imshow('res', res)
        myWorker.frame_rgb=myWaI.draw_grasp_point_and_arm_center(myWorker.frame_rgb,myWorker.grasp_point)
        cv2.imshow('rgb', myWorker.frame_rgb)
        # cv2.imshow('depth_gray', mySS.frame_depth_gray)
        cv2.imshow('depth_rgb', myWorker.frame_depth_rgb)
        flag = cv2.waitKey(30) & 0xFF
        # out.write(frame)
        if flag == 27:
            rospy.signal_shutdown("User hit ESC key to quit.")
            break
        elif flag == 32:
            # 空格记录数据

            pass


def run():
    # 检查1.抓取完成2.摆放
    print('----------Assist----------')
    while True:
        sleep(0.2)
        if  myWorker.wait_or_act == False:
            # 预测参数，并发送，反转控制器
            if myWorker.grasp_point[2] != 0:
                myWorker.target = myWorker.predict(myWorker.coor)  # 存上参数
                actions=myWorker.execute_a_set_of_action(myWorker.target)
                for a in actions:
                    myContact.writeSCM(a)
                    if myContact.readStatu():
                        pass
                myWorker.wait_or_act = not myWorker.wait_or_act#开启等待状态
    pass


if __name__ == '__main__':
    # q = Queue(1)
    cb = CvBridge()
    myContact = Communicate_with_SCM()
    # 开启线程
    # mythread = Thread(target=run)
    # mythread.daemon=True
    # mythread.start()
    # instance
    myWorker = Worker()
    myWaI = contour.Where_am_I()
    # 3.rosnode
    # 3 开始工作，订阅和初始化一些参数
    node_name = 'jurvis_worker'
    rospy.init_node(node_name)
    rospy.on_shutdown(clean)
    properity(480, 640)
    rospy.init_node(node_name)
    image_sub = rospy.Subscriber("/camera/color/image_raw", Image, myWorker.convert_Color, buff_size=2097152)  # 2MB
    depth_sub = rospy.Subscriber("/camera/depth/image_rect_raw", Image, myWorker.convert_Depth, buff_size=2097152)
    try:
        main()
    except KeyboardInterrupt:
        print 'end'
        cv2.destroyAllWindows()
