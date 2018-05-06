# coding=utf-8
import numpy as np
from calibrate import Communicate_with_SCM
from time import sleep,time
input=raw_input
myCS=Communicate_with_SCM()
actions=np.loadtxt('./data_base/standard_action.txt',delimiter=',')

actions=actions.astype(np.int32)

actions=actions.tolist()
# print(actions,type(actions),type(actions[0]))
# exit()
kind=[0,1,0,0,1,0]
num=[0]*len(kind)
cave=900
setPosi=[400,600]
init_elevate=1750
elevate = init_elevate
sleep(1)
for i,target in enumerate(actions):
    print(i)
    # 先执行一套动作
    motions = [
                  [1200, 600, 1900, 1000, 1500, target[3], 800],  # 到达上面
                  [1200, 600] + target + [500],  # 到达
                  [1750, 600] + target + [50],  # 合上爪子
                  [1750, 600, 1900, 1000, 1600, target[3], 450],  # 抬高
                  [1750, 600, 1900, 1000, 1600, cave, 400],  # 到桶上
                  [1750, 2500, 1900, 1000, 1600, cave, 500],  # 倾倒
                  # [1750, 2500, 1800, 1000, 1500, cave, 20],  # 倾倒-抖一下
                  # [1750, 2500, 2000, 1000, 1500, cave, 20],  # 倾倒-抖一下
                  [1750, 600, 1900, 1000, 1500, setPosi[kind[i]], 800],  # 转回
                  [1400, 600, 1700, 800,elevate, setPosi[kind[i]], 700],  # 摆放
                  [1300, 600, 1600, 800,1500, setPosi[kind[i]], 300],  # 摆放
                  [1200, 600, 1200, 1000, 1350, setPosi[kind[i]], 300]  # 归位
              ]
    # 然后看看要不要推开
    num[kind[i]]+=1
    if num[kind[i]] < 3:
        elevate = init_elevate - num[kind[i]] * 140
    else:
        # 到了5只碗高，推一下
        elevate=init_elevate
        num[kind[i]]=0
        motions0 = [
            [1300,1600,1300, 850, 1650, setPosi[kind[i]], 400],  # 转回
            [1400, 1600, 2100,1000,2000, setPosi[kind[i]], 500],  # 推出
            [1200, 600, 1200, 1000, 1350, setPosi[kind[i]], 300]  # 归位
        ]
        motions = motions + motions0

    for a in motions:
        myCS.writeSCM(a)
        # g=input("输入任意东西")
        if myCS.readStatu():
            pass
    if i in [1,4]:
        t=input("wait next batch:")