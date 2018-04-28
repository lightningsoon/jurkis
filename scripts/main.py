#!/usr/bin/env python
# -*- coding=utf-8 -*-

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
from Program.Detection.detect import minor,properity
from Program.Calibration.calibrate import Sense_Self,ros_spinOnce,Communicate_with_SCM
from Program.Calibration.EyeArm_Learning import Pinocchio
import Queue
from time import sleep
from threading import Thread
fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('output.mp4', fourcc, 30.0, (640, 480))


class Worker(Sense_Self,Pinocchio):
    def __init__(self):
        super(Worker, self).__init__()
        #TODO 应该要给True
        self.wait_or_act=False#True 等待参数，False正在行动
        self.target=None
        self.kind=None#种类名字
        self.num={}#每种数量
        self.__posi={}#每种位置
        # self.coor[:2] # grasp point
        # TODO 填写位置（未完成）
        self.__setPosi=[[1500,600,1616,1200,1850,800],[1500,600,1616,1200,1850,700],[1500,600,1616,1200,1850,600]]
        pass
    def convert_Color(self,img):
        # 目标检测
        try:
            img = cb.imgmsg_to_cv2(img, 'bgr8')
        except CvBridgeError, e:
            print e
        self.frame_rgb=np.array(img, dtype=np.uint8)
    def working(self):
        '''
        工作部分，接在图像转格式存储后面
        :return:
        '''
        result,self.__x[:2]=minor(self.frame_rgb)
        self.coor=self.__x#把数据带到外面来
        return result
        pass
    def execute_a_set_of_action(self):
        '''
        这次就手动设置
        #TODO 以后依靠识别，配合有监督示范
        :param self.target:
        :return:
        '''

        # 打开爪子

        # 去
        # 夹住
        # 抬高
        # 到桶的上方
        # 倾倒
        # 摆放
        # 收工
        pass
    def stack(self):
        # TODO 搬到合适点位置（未完成） 计数
        if self.kind in self.num.keys():
            self.num[self.kind] += 1
            # TODO 每个要移动高一点，算一下，太高了要重新选位置（未完成）
            seat=self.__posi[self.kind]
            Communicate_with_SCM.writeSCM(seat)
        else:
            # 第一次见到，计数归零，分配位置
            self.num[self.kind]=0
            self.__posi[self.kind]=self.__getPosi()
        pass
    def __getPosi(self):
        for x in self.__setPosi:
            yield x
        print("!!!!!! Position is not enough  !!!!!! ")
        exit()
    pass
def clean():
    cv2.destroyAllWindows()
    print 'shut down'

def main():
    print('start')
    while True:
        ros_spinOnce()
        res=myWorker.working()
        cv2.imshow('res',res)
        cv2.imshow('rgb', myWorker.frame_rgb)
        # cv2.imshow('depth_gray', mySS.frame_depth_gray)
        cv2.imshow('depth_rgb', myWorker.frame_depth_rgb)
        flag = cv2.waitKey(30) & 0xFF
        # out.write(frame)
        if flag == 27:
            rospy.signal_shutdown("User hit ESC key to quit.")
            break
        elif flag == 32:
            # 记录数据
            # mySS.flag4can_put_coor = True
            pass
        if myWorker.coor[:2] != [None, None] and myWorker.wait_or_act:
            # 检测到数据，预测参数，并发送，反转控制器
            myWorker.coor[2] = myWorker.getDeepth()
            myWorker.target = myWorker.predict(myWorker.coor)#存上参数
            # myWorker.execute_a_set_of_action(myWorker.param)
            myWorker.wait_or_act=not myWorker.wait_or_act
def run():
    # 检查1.抓取完成2.摆放
    print('----------Assist----------')
    while True:
        sleep(0.2)
        if myWorker.wait_or_act==False:
            myWorker.execute_a_set_of_action()
        if myContact.readStatu():

            while True:
                sleep(0.1)
                if myContact.readStatu():
                    # 第二次读到1 转回grasp状态
                    myWorker.wait_or_act=not myWorker.wait_or_act
    pass

if __name__ == '__main__':
    # q = Queue(1)
    cb = CvBridge()
    myContact=Communicate_with_SCM()
    # 开启线程
    # mythread = Thread(target=run)
    # mythread.daemon=True
    # mythread.start()
    # instance
    myWorker=Worker()
    # rosnode
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