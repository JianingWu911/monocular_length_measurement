# import cv2
# import numpy as np

# # 载入视频
# cap = cv2.VideoCapture('test.MP4')

# while True:
#     # 读取视频帧
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # 将帧转换为HSV颜色空间
#     hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

#     # 设置红色范围
#     lower_red = np.array([0, 100, 100])  # 最低红色阈值
#     upper_red = np.array([10, 255, 255])  # 最高红色阈值
#     mask = cv2.inRange(hsv_frame, lower_red, upper_red)

#     # 提取轮廓
#     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
#     # 遍历轮廓
#     for contour in contours:
#         # 计算轮廓的圆心和半径
#         (x, y), radius = cv2.minEnclosingCircle(contour)
#         center = (int(x), int(y))
#         radius = int(radius)
        
#         # 绘制圆心和轮廓
#         cv2.circle(frame, center, radius, (0, 255, 0), 2)
#         cv2.circle(frame, center, 2, (0, 0, 255), 3)
        
#     # 显示帧
#     cv2.imshow('Frame', frame)
    
#     # 按下'q'键退出
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # 释放视频和关闭窗口
# cap.release()
# cv2.destroyAllWindows()

import cv2
import numpy as np

# 定义基准长度
stand_dis = 50 # mm
dis_pix_ratio = 0

# 载入图片
image = cv2.imread('oneshot.jpg')
# 绘制轮廓
def draw_contour(contour):
    # 遍历轮廓
    for contour in contours:
        # 计算轮廓的外接圆
        (x, y), radius = cv2.minEnclosingCircle(contour)
        center = (int(x), int(y))
        radius = int(radius)
        
        # 绘制轮廓的外接圆
        cv2.circle(image, center, radius, (0, 255, 0), 10)
        
        # 绘制圆心
        cv2.circle(image, center, 5, (0, 0, 255), -1)
        

# 将图片转换为HSV格式
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 红色区域的HSV范围
lower_red = np.array([0, 100, 100])
upper_red = np.array([10, 255, 255])
mask1 = cv2.inRange(hsv, lower_red, upper_red)

lower_red = np.array([160, 100, 100])
upper_red = np.array([179, 255, 255])
mask2 = cv2.inRange(hsv, lower_red, upper_red)

# 合并两个遮罩
mask = cv2.bitwise_or(mask1, mask2)
mask = cv2.medianBlur(mask, 5)

# cv2.imshow('Mask', mask)
# cv2.waitKey(0)


# 定义蓝色的HSV范围
lower_blue = np.array([100, 50, 50])
upper_blue = np.array([130, 255, 255])

# 生成蓝色区域的遮罩
mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

# 对遮罩进行中值滤波以去噪
mask_blue = cv2.medianBlur(mask_blue, 5)

# 在原图上绘制蓝色区域
# cv2.imshow('Mask_blue', mask_blue)
# cv2.waitKey(0)


# 提取蓝色轮廓
contours, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
draw_contour(contours)

# 计算蓝色圆心间的像素距离和距离像素比
if len(contours) >= 2:
    center1 = np.array(contours[0][0][0])
    center2 = np.array(contours[1][0][0])
    distance = np.linalg.norm(center1 - center2)
    dis_pix_ratio = stand_dis / distance
    print("圆心间的距离为：", distance)
    print("距离像素比--dis_pix_ratio", dis_pix_ratio)

# 提取红色轮廓
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
draw_contour(contours)

# 计算红色圆心间像素距离和实际距离
if len(contours) >= 2:
    center1 = np.array(contours[0][0][0])
    center2 = np.array(contours[1][0][0])
    distance = np.linalg.norm(center1 - center2)
    print("圆心间的像素距离：", distance)
    print("圆心间的实际距离：", distance * dis_pix_ratio)

# 显示处理后的图片
cv2.imshow('Processed Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
