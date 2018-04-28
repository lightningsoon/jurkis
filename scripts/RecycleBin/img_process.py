# -*- coding=utf-8 -*-
'''
尝试过点图像处理方法
部分
'''
print('以下代码仅供留念，不可运行')
exit()

# 蓝色或红色hsv值
self.mask_low = np.array([28, 0, 230])
self.mask_high = np.array([35, 100, 255])
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
# 根据阈值构建掩模
res = cv2.inRange(hsv, self.mask_low, self.mask_high)
self.debug_img = res
# res=cv2.cvtColor(cv2.cvtColor(mask,cv2.COLOR_HSV2BGR),cv2.COLOR_BGR2GRAY)
# print(mask.shape)
# 阈值一定要设为 0！
ret, res = cv2.threshold(res, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
res = cv2.Canny(res, 50, 150, apertureSize=3)

# TODO 调参
circles = cv2.HoughCircles(res, cv2.HOUGH_GRADIENT, dp=1, minDist=8,
                           param1=10, param2=10, minRadius=2, maxRadius=6)

circles = circles[0, 0]
circles = np.uint16(np.around(circles))[:2]