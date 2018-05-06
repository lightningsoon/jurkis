#!/usr/bin/env python
# -*- coding=utf-8 -*-
import traceback
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
# fourcc = cv2.VideoWriter_fourcc(*'XVID')


# out = cv2.VideoWriter('output.mp4', fourcc, 30.0, (640, 480))


class Worker(Sense_Self, Pinocchio):
    def __init__(self):
        # for i in range(len(self.__class__.mro())):
        # 多继承初始化
        super(Worker, self).__init__()

        # 应该要给True
        self.wait_or_act = True  # True 等待参数，False正在行动
        self.grasp_point=[None]*3
        self.target = None
        self.kind = None  # 种类名字
        self.num = [0]*5  # 每种数量
        # TODO 填写可以摆放的位置
        self.__setPo5si = [850,750,650,550,500]
        # exit()
        pass

    def convert_Color(self, img):
        try:
            img = cb.imgmsg_to_cv2(img, 'bgr8')
        except CvBridgeError as e:
            print e
        # print(self.frame_rgb)
        self.frame_rgb = np.array(img, dtype=np.uint8)

    def __getDeepth(self):
        '''
        抓取点点深度
        :return:
        '''
        return int(self._frame_depth_raw[self.grasp_point[1],self.grasp_point[0]])
    def working(self):
        '''
        5 工作部分，接在图像转格式存储后面
        :return:
        '''
        self.frame_rgb, self.grasp_point[:2],self.kind = minor(self.frame_rgb,self.wait_or_act)# 把数据带到外面来
        if self.grasp_point[:2] != [None,None] and myWorker.wait_or_act==True :#可以执行并且，机械臂等待中
            self.grasp_point[2]=self.__getDeepth()
            if self.grasp_point[2]!=0:
                myWorker.wait_or_act=False#开始让子线程工作
            else:
                print("深度为0")
        # return result
        pass

    def execute_a_set_of_action(self, target):
        '''
        这次就手动设置

        :param target:[2:6]
        :return:
        '''
        cave=1000# TODO 坑的位置
        # 统计一下数量情况
        # print(self.kind)
        # 修正
        target[2]=target[2]+100#越小越高
        if target[1]<400:
            target[1]=target[1]+1000
        elif target[1]<900:
            target[1]=target[1]+300
        elif target[1]<1100:
            target[1]=int(target[1]*1.12)
        target[0]+=0#越大越高
        if self.num[self.kind]<3:
            elevate=1900-self.num[self.kind]*100
            self.num[self.kind]+=1
            # TODO 以后依靠识别，配合有监督示范
            # TODO 以后在初始化时，写到下位机
            actions0 = []
        else:
            #到了5只碗高，推一下
            elevate=1900
            actions0=[
                [1200, 1600, 1300, 1000, elevate, self.__setPosi[self.kind], 400],  # 转回
                [1200, 1600, 1700, 1200, elevate, self.__setPosi[self.kind], 400],  # 推出
                [1200, 1600, 1200, 1000, 1350, self.__setPosi[self.kind], 300]  # 归位
            ]
        actions=actions0+\
                [
                    [1200, 600, 1900, 1000, 1500,target[3], 600],  # 到达
                    [1200, 600] + target + [700],  # 到达
                    [1700, 600] + target + [50],  # 合上爪子
                    [1700, 600, 1900, 1000, 1500, target[3], 800],  # 抬高
                    [1700, 600, 1900, 1000, 1500, cave, 600],  # 到桶上
                    [1700, 2500, 1900, 1000, 1500, cave, 500],  # 倾倒
                    # [1700, 2500, 1800, 1000, 1500, cave, 20],  # 倾倒-抖一下
                    # [1700, 2500, 2000, 1000, 1500, cave, 20],  # 倾倒-抖一下
                    [1700, 600, 1900, 1000, elevate, self.__setPosi[self.kind], 800],  # 转回
                    [1700, 600, 1900, 1000, elevate, self.__setPosi[self.kind], 400],  # 下落
                    [1300, 600, 1900, 1000, elevate, self.__setPosi[self.kind], 200],  # 摆放
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
    myWorker.waitRosMsg()
    while True:
        # 4 获取图像
        ros_spinOnce()
        # 5 处理数据
        myWorker.working()
        # cv2.imshow('res', res)
        myWorker.frame_rgb=myWaI.draw_grasp_point_and_arm_center(myWorker.frame_rgb,myWorker.grasp_point[:2])
        cv2.imshow('rgb', myWorker.frame_rgb)
        # cv2.imshow('depth_gray', mySS.frame_depth_gray)
        # cv2.imshow('depth_rgb', myWorker.frame_depth_rgb)
        flag = cv2.waitKey(30) & 0xFF
        # out.write(frame)
        if flag == 27:
            rospy.signal_shutdown("User hit ESC key to quit.")
            break
        elif flag == 32:
            # 空格记录数据

            pass
    exit()
def run():
    # 检查1.抓取完成2.摆放
    print('----------Assist----------')
    while True:
        try:
            sleep(0.2)
            myWorker
        except TypeError:
            print("程序退出！")
            exit()
            # raise TypeError
        if  myWorker.wait_or_act == False:
            # 预测参数，并发送，反转控制器
            print("point",myWorker.grasp_point)
            if not myWorker.grasp_point[:2]==[None,None]:
                myWorker.target = myWorker.predict(myWorker.grasp_point)  # 存上参数
                print("target",myWorker.target)
                if np.any(np.array(myWorker.target)<0):
                    print("faultly predict",myWorker.target)
                    pass
                else:
                    actions=myWorker.execute_a_set_of_action(myWorker.target)
                    for a in actions:
                        myContact.writeSCM(a)
                        # g=input("输入任意东西")
                        if myContact.readStatu():
                            pass
            myWorker.wait_or_act = not myWorker.wait_or_act#开启等待状态
    pass


if __name__ == '__main__':
    # q = Queue(1)
    cb = CvBridge()
    myContact = Communicate_with_SCM()
    myWorker = Worker()
    myWaI = contour.Where_am_I()
    # 开启线程
    mythread = Thread(target=run)
    mythread.daemon = True
    mythread.start()
    # 3.rosnode
    # 3 开始工作，订阅和初始化一些参数
    node_name = 'jurkis_worker'
    rospy.init_node(node_name)
    rospy.on_shutdown(clean)
    properity(480, 640)
    rospy.init_node(node_name)
    image_sub = rospy.Subscriber("/camera/color/image_raw", Image, myWorker.convert_Color, buff_size=2097152)  # 2MB
    depth_sub = rospy.Subscriber("/camera/depth/image_rect_raw", Image, myWorker.convert_Depth, buff_size=2097152)
    main()
    '''
    try:
        main()
    except Exception as e:
        print '小哥哥，程序挂了，来修我啊！'
        traceback.format_exc(e)
        cv2.destroyAllWindows()
        exit()
    '''