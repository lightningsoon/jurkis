#!/usr/bin/env python
# -*- coding=utf-8 -*-

from time import sleep
from calibrate import Communicate_with_SCM
from calibrate import ArmEye_collectingData
import logging

myCS = Communicate_with_SCM()
myAL = ArmEye_collectingData()

def main():
    sleep(2)
    for p in myAL.para:
        myCS.writeSCM(p)
        # 等待到达
        if (myCS.readStatu()):
            print(p)
            pass
        pass
    myCS.home_arm()
    logging.warn('exhaust parameters')

try:
    main()
except KeyboardInterrupt:
    myCS.home_arm()
    myCS.closeSerial()
# myCS.home_arm([1468,600,1100,900,1750,2000])