from cv2 import aruco
import cv2
import random
import numpy as np

marker_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_50)


def geneAruco():
    A4 = (2100,2940)
    A4paper=np.ones(A4,np.uint8)*255
    x,y=10,10
    for i in range(50):
        size=10*i+80
        img=aruco.drawMarker(marker_dict,i,size)
        y0,x0=y+size,x+size
        if x0>A4[1]:
            x=10
            y=y0+20
            y0, x0 = y + size , x + size
        if y0 > A4[0]:
            break
        try:
            A4paper[y:y0,x:x0]=img
        except ValueError:
            print(y,y0,i,size,x,x0)
            flag=cv2.waitKey(0)
            if flag == 27:
                break
        x=x0+20
        cv2.imshow('',cv2.resize(A4paper,(0,0),None,0.3,0.3,cv2.INTER_LINEAR))
        flag=cv2.waitKey(10)
        if flag==27:
            break
    print(i)
    cv2.imwrite('aruco.png',A4paper)

# geneAruco()
def detectMarker():
    cap=cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
    while True:
        _,frame=cap.read()
        corners, ids, rejectedImgPoints = aruco.detectMarkers(frame, marker_dict)
        frame = aruco.drawDetectedMarkers(frame, corners, ids)
        cv2.imshow('??', frame)
        flag=cv2.waitKey(30)
        if flag==27:
            break
# img=cv2.imread('aruco.png')
# corners, ids, rejectedImgPoints=aruco.detectMarkers(img,marker_dict)
# img=aruco.drawDetectedMarkers(img,corners,ids)
# # img=cv2.resize(img,(0,0),None,0.3,0.3,cv2.INTER_LINEAR)
# cv2.imshow('??',img)
# cv2.waitKey(0)
detectMarker()