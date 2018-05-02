#!/usr/bin/env python
# -*- coding=utf-8 -*-
import os
# 帮助加载一下模型
print(os.path.abspath('.'))
def main():
    # 1.先计算旋转中心
    flag = os.system('rosrun jurvis find_arm_position.py')
    assert flag==0
    # 预先得到分类中心？暂不考虑
    # flag = os.system("python ./Program/Outline/cluster.py")
    # assert flag==0
    # 2.开始工作
    flag = os.system('rosrun jurvis main.py')
    assert flag==0
    pass
if __name__ == '__main__':
    main()
