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
        self.num = {}  # 每种数量
        self.__posi = {}  # 每种位置
        # TODO 填写可以摆放的位置（未完成）
        self.__setPosi = [1000,800,600,400]
        pass

    def convert_Color(self, img):
        # 目标检测
        try:
            img = cb.imgmsg_to_cv2(img, 'bgr8')
        except CvBridgeError, e:
            print e
        self.frame_rgb = np.array(img, dtype=np.uint8)

    def working(self):
        '''
        工作部分，接在图像转格式存储后面
        :return:
        '''
        result, self.__x[:2],self.kind = minor(self.frame_rgb)
        self.grasp_point = self.__x  # 把数据带到外面来
        return result
        pass

    def execute_a_set_of_action(self, target):
        '''
        这次就手动设置

        :param target:[2:6]
        :return:
        '''
        # TODO 以后依靠识别，配合有监督示范
        # TODO 坑的位置
        cave=1000
        actions = [
            [1200, 600 ] + target + [600],  # 到达
            [1700, 600]+target + [50],  # 合上爪子
            [1700, 600, 1900, 1000, 1500, target[3], 500],  # 抬高
            [1700, 600, 1900, 1000, 1500, cave, 700],  # 到桶上
            [1700, 2500, 1900, 1000, 1500, cave, 500],  # 倾倒
            # [1700, 2500, 1800, 1000, 1500, cave, 20],  # 倾倒-抖一下
            # [1700, 2500, 2000, 1000, 1500, cave, 20],  # 倾倒-抖一下
            [1700, 600, 1900, 1000, 1500, 1200, 400],  # 转回
            [1700, 600, 1900, 1000, 1950, 1200, 800],  # 下落
            [1300, 600, 1900, 1000, 1950, 1200],  # 摆放
            [1200, 600, 1100, 1000, 1350, 1200, 600]  # 归位
        ]
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
            seat = self.__posi[self.kind]
            Communicate_with_SCM.writeSCM(seat)
        else:
            # 第一次见到，计数归零，分配位置
            self.num[self.kind] = 0
            self.__posi[self.kind] = self.__getPosi()
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
        res = myWorker.working()
        cv2.imshow('res', res)
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
        if myWorker.grasp_point[:2] != [None, None] and myWorker.wait_or_act == True:
            # 检测到数据&机械臂正在等待
            # 预测参数，并发送，反转控制器
            myWorker.wait_or_act = not myWorker.wait_or_act  # 关闭预测
            myWorker.grasp_point[2] = myWorker.getDeepth()
            if myWorker.grasp_point[2] != 0:
                myWorker.target = myWorker.predict(myWorker.coor)  # 存上参数
                myWorker.execute_a_set_of_action(myWorker.target)
                if myContact.readStatu():
                    myWorker.wait_or_act = not myWorker.wait_or_act
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
