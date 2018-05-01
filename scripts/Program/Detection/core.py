# -*- coding:utf-8 -*-
'''
using opencv to implement visualization
'''
import cv2
import numpy as np
import logging
from ..Outline import contour,cluster


# a=list(map
#        (tuple,np.random.randint(0,255,(100,3))))
# for i in range(25):
#     print('%s, %s, %s, %s,' % (a[i*4],a[i*4+1],a[i*4+2],a[i*4+3]))
color_map = [(226, 230, 224), (74, 123, 60), (1, 112, 192), (122, 103, 253),
                 (20, 242, 6), (247, 28, 99), (161, 91, 62), (85, 78, 178),
                 (63, 46, 225), (12, 141, 224), (115, 59, 49), (250, 207, 28),
                 (100, 127, 246), (161, 239, 82), (189, 98, 219), (78, 139, 192),
                 (77, 133, 61), (81, 110, 229), (132, 65, 153), (253, 215, 89),
                 (154, 59, 157), (144, 236, 83), (206, 162, 250), (68, 45, 148),
                 (89, 70, 91), (80, 182, 94), (228, 180, 161), (97, 155, 35),
                 (224, 43, 45), (85, 4, 58), (166, 86, 121), (44, 164, 35),
                 (247, 135, 240), (127, 184, 149), (82, 81, 253), (221, 86, 57),
                 (227, 189, 169), (8, 140, 109), (234, 39, 60), (105, 119, 241),
                 (189, 8, 114), (156, 66, 223), (123, 73, 210), (238, 9, 62),
                 (219, 107, 253), (208, 138, 243), (178, 53, 126), (186, 75, 66),
                 (114, 206, 193), (176, 56, 167), (168, 145, 197), (119, 242, 14),
                 (119, 77, 98), (96, 251, 221), (27, 20, 71), (131, 104, 170),
                 (144, 245, 225), (169, 173, 177), (249, 39, 127), (237, 227, 168),
                 (152, 22, 34), (148, 115, 46), (158, 102, 58), (33, 83, 139),
                 (174, 36, 131), (111, 44, 37), (14, 172, 155), (232, 232, 231),
                 (147, 188, 97), (34, 21, 145), (247, 124, 199), (91, 97, 18),
                 (97, 229, 215), (81, 15, 65), (57, 188, 4), (174, 51, 182),
                 (87, 206, 151), (11, 134, 14), (126, 98, 201), (243, 152, 106),
                 (247, 71, 189), (108, 213, 83), (115, 206, 170), (58, 178, 16),
                 (59, 155, 48), (100, 7, 204), (109, 29, 200), (202, 99, 41),
                 (73, 215, 32), (185, 160, 164), (20, 166, 39), (133, 213, 56),
                 (155, 47, 247), (13, 172, 43), (117, 159, 238), (28, 105, 185),
                 (187, 173, 26), (208, 79, 215), (92, 34, 138), (107, 230, 217)]  # 100个

def constant4ros(category_index, h=480, w=640):
    global height, width, color_map, kind_name,j,reference_point
    height, width = h, w
    kind_name=category_index
    j = 0#截图计数
    reference_point = width // 2, height


def constant(image, category_index, cls_num=None):
    global height, width, color_map, kind_name,reference_point
    kind_name = category_index
    # print(kind_name)
    height, width, _ = image.shape
    reference_point = width // 2, height



def master(image, num, boxs, classes, scores, max_boxes_to_draw=10, min_score_thresh=.5):
    # 5.1 数据整理
    global height, width, color_map, kind_name
    # 47cup,50spoon,51bowl
    if num>0:
        cup_indices = np.argwhere(classes == 47)
        bowl_indices = np.argwhere(classes == 51)
        # 合并
        indices1=np.append(cup_indices,bowl_indices)
        indices2=np.argwhere((scores>min_score_thresh)==True).ravel()
        indices=set(indices1)&set(indices2)
        #根据分值和类别得到索引
        boxs=boxs[indices][:max_boxes_to_draw]
        scores=scores[indices][:max_boxes_to_draw]
        # 6、7 圈点框，找和画
        what=mixFuc_crucial(image,boxs,scores,max_boxes_to_draw)
    else:
        what=(None,None),None
    return what
    pass

myWhere_am_I=contour.Where_am_I()
myWho_am_I=cluster.Who_am_I()
def mixFuc_crucial(image,boxs,scores,length):
    '''
    这个函数非常重要!!
    而且目前无法拆分，这样效率最高^o^
    循环只有一个
    内部数据以复制为主
    它有
        修正图像边框值
        绘制边框
        放上
        画圆
        计算抓取点
        绘制点
    # TODO 这个函数可以用DNN代替，直接获得抓取位置
    :param image:
    :param boxs:
    :param scores:
    :param length:
    :return:可能是接下来要抓取点（二维），也可能没有可以抓的点
    '''

    global height, width,j,color_map
    img_raw = image.copy()
    grasp_points,new_indices,grasp_dists=[],[],[]#抓取点，有圈的索引，抓取距离
    for i in range(length):
        ##############

        ymin, xmin, ymax, xmax = list(map(int, (boxs[i][0] * height, boxs[i][1] * width,
                                                boxs[i][2] * height, boxs[i][3] * width)))

        ##############
        y0, y1, x0, x1 = max(0, ymin - 5), min(480, ymax + 5), \
                         max(0, xmin - 5), min(640, xmax + 5)
        img_raw_mini=img_raw[y0:y1, x0:x1]
        # 7.1 找+画 圈
        image[y0:y1, x0:x1], circle_point = contour.circle(img_raw_mini,
                                                           image[y0:y1, x0:x1])
        #有圈
        if circle_point:
            new_indices.append(i)#记录有圆圈点索引
            temp=myWhere_am_I.calculate_grasp_point(circle_point)#return第一个是点，第二个距离
            grasp_points.append(temp[0])#记录最近的点
            grasp_dists.append(temp[1])
        # 7.2 画框
        rectangle(image,ymin, xmin, ymax, xmax,scores[i],color_map[i])
    # 看哪个圈的点适合抓取
    if len(new_indices) != 0:
        min_grasp_dist_index = np.argmin(grasp_dists).ravel()[0]
        # 7.2 查看当前框的种类
        temp_indc=new_indices[min_grasp_dist_index]
        ymin, xmin, ymax, xmax = list(map(int, (boxs[temp_indc][0] * height, boxs[temp_indc][1] * width,
                                                boxs[temp_indc][2] * height, boxs[temp_indc][3] * width)))
        y0, y1, x0, x1 = max(0, ymin - 5), min(480, ymax + 5), \
                         max(0, xmin - 5), min(640, xmax + 5)
        img_raw_mini=img_raw[y0:y1, x0:x1]
        temp_kind=myWho_am_I.get_kind(img_raw_mini)
        # cv2.drawMarker(image, tuple(grasp_points[min_grasp_dist_index]), (0, 0, 255), cv2.MARKER_STAR, 10, 2)
        return grasp_points[min_grasp_dist_index],temp_kind
    return (None,None),None

def rectangle(image,ymin, xmin, ymax, xmax,score_i,color_map_i):
    # 画框
    cv2.rectangle(image,
                  (xmin, ymin),
                  (xmax, ymax),
                  color_map_i, 3, cv2.LINE_AA)
    cv2.putText(image, str(score_i), (xmin, ymin),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, 8)

'''
def draw1Pic(num, indices, scores, min_score_thresh, boxs, image, class_index):
    from ...Outline import contour
    global height, width,j
    img_raw=image.copy()
    indices = indices[:min(num, len(indices))].ravel()
    grasp_points,new_indices,grasp_dists=[],[],[]
    # TODO D2 下面两行可以改进成去除评分小的点，或者设置为false，
    for i in indices:
        if scores[i] < min_score_thresh:
            # logging.warning('score is too low!')
            continue
        # print(boxs[i])
        ymin, xmin, ymax, xmax = list(map(int, (boxs[i][0] * height, boxs[i][1] * width,
                                                boxs[i][2] * height, boxs[i][3] * width)))
        # TODO D1 圈@鲍鲍，点@莫名
        # //处理原始图（参数一），但是在image上画内容，否则会有重叠画图干扰
        y0,y1,x0,x1=max(0,ymin-5),min(480,ymax+5),max(0,xmin-5),min(640,xmax+5)
        image[y0:y1,x0:x1],circle_point=contour.circle(img_raw[y0:y1,x0:x1],
                                                  image[y0:y1,x0:x1])
        # 有圆点的不返回None
        if circle_point:
            new_indices.append(i)#记录有圆圈点索引
            temp=contour.calculate_grasp_point(circle_point)#第一个是点，第二个距离
            grasp_points.append(temp[0])#记录最近的点
            grasp_dists.append(temp[1])

        # TODO 比较并记录最近点??（未完成）

        # 自动截图
        # 问题：可能会截图到上一张被覆盖的文字或框
        # 方案：1复制一张图像，使用原始图；2使用该功能关闭另外两个；
        # if j%10==0:
        #     cv2.imwrite('/home/momo/catkin_ws/src/jurvis/scripts/Program/Outline/data/'+
        #                 str(j//10)+'.png',image[y0:y1,x0:x1])
        # j += 1

        # 框+颜色
        cv2.rectangle(image,
                      (xmin, ymin),
                      (xmax, ymax),
                      color_map[class_index], 3, cv2.LINE_AA)

        # 类别+得分
        try:
            class_name = kind_name[class_index]['name']
        except KeyError:
            class_name = 'N/A'
            # logging.warning('no class_name!')
        cv2.putText(image, class_name + ':' + str(scores[i]), (xmin, ymin),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, 8)
    # 看哪个圈点点适合抓取
    if len(new_indices)!=0:
        min_grasp_dist_index=np.argmin(grasp_dists)
        cv2.drawMarker(image,grasp_points[min_grasp_dist_index],(0,0,255),cv2.MARKER_STAR,10,2)
        return new_indices[min_grasp_dist_index]
    return None

'''
