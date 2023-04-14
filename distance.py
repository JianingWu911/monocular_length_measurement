import cv2
import numpy as np
import matplotlib.pyplot as plt

# 定义基准长度
stand_dis = 50 # mm
dis_pix_ratio = 0
dis_meas = []
# 绘制轮廓函数
def draw_contour(contours):
    cv2.drawContours(image, contours, -1, (0, 0, 0), 5)
    # 遍历轮廓
    for contour in contours:
        # 计算轮廓的外接圆
        (x, y), radius = cv2.minEnclosingCircle(contour)
        center = (int(x), int(y))
        radius = int(radius)
        
        # 绘制轮廓的外接圆
        # cv2.circle(image, center, radius, (0, 255, 0), 2)
        
        # 绘制圆心
        # cv2.circle(image, center, 2, (0, 0, 255), -1)

# 调试阈值用
def empty(i):
    pass

def choose_contours(contours) :
    # 设置面积阈值
    area_threshold = 500  # 可根据需求调整

    # 新建一个列表，用于存储面积大于阈值的轮廓
    filtered_contours = []

    # 遍历找到的轮廓
    for contour in contours:
        # 计算轮廓的面积
        area = cv2.contourArea(contour)
        # 如果面积大于阈值，则将轮廓加入筛选后的轮廓列表中
        if area > area_threshold:
            filtered_contours.append(contour)
        
    return filtered_contours

# 载入视频
cap = cv2.VideoCapture('test.MP4')

# # 调试用代码找到合适的阈值，用来产生控制滑条
# cv2.namedWindow("HSV")
# cv2.resizeWindow("HSV", 640, 300)
# cv2.createTrackbar("HUE Min", "HSV", 0, 179, empty)
# cv2.createTrackbar("SAT Min", "HSV", 0, 255, empty)
# cv2.createTrackbar("VALUE Min", "HSV", 230, 255, empty)
# cv2.createTrackbar("HUE Max", "HSV", 78, 179, empty)
# cv2.createTrackbar("SAT Max", "HSV", 190, 255, empty)
# cv2.createTrackbar("VALUE Max", "HSV", 255, 255, empty)

# # 帧率计数器
# frame_cnt = 0
# N = 100

while True:
    # 读取视频帧
    ret, frame = cap.read()

    # # 帧计数器
    # frame_cnt += 1
    # if (frame_cnt % N != 1) :
    #     continue

    # 如果没有轮廓 终止
    if not ret:
        # 绘制曲线

        plt.plot(dis_meas)

        # 设置 x 轴和 y 轴的标签
        plt.xlabel('帧')
        plt.ylabel('测量长度')

        # 设置图像标题
        plt.title("长度测量值")

        # 显示图像
        plt.show()
        print("没有输入，结束")
        break
    
    # 将帧转换为HSV颜色空间 image 为每一帧的HSV图像
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # # 调试用代码找到合适的阈值，用来产生控制滑条
    # h_min = cv2.getTrackbarPos("HUE Min", "HSV")
    # h_max = cv2.getTrackbarPos("HUE Max", "HSV")
    # s_min = cv2.getTrackbarPos("SAT Min", "HSV")
    # s_max = cv2.getTrackbarPos("SAT Max", "HSV")
    # v_min = cv2.getTrackbarPos("VALUE Min", "HSV")
    # v_max = cv2.getTrackbarPos("VALUE Max", "HSV")
    
    # lower = np.array([h_min, s_min, v_min])
    # upper = np.array([h_max, s_max, v_max])

    # 将图片转换为HSV格式
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 定义蓝色的HSV范围
    lower_blue = np.array([44, 63, 86])
    upper_blue = np.array([81, 210, 209])

    # 生成蓝色区域的遮罩
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    # 对遮罩进行中值滤波以去噪
    mask_blue = cv2.medianBlur(mask_blue, 5)

    # 在原图上绘制蓝色区域
    cv2.imshow('Mask_blue', mask_blue)
    cv2.waitKey(100)


    # 提取蓝色轮廓
    contours, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours_blue = choose_contours(contours)
    # print(len(filtered_contours_blue))
    draw_contour(filtered_contours_blue)

    # 计算蓝色圆心间的像素距离和距离像素比
    if len(filtered_contours_blue) >= 2:
        center1 = np.array(filtered_contours_blue[0][0][0])
        center2 = np.array(filtered_contours_blue[1][0][0])
        distance = np.linalg.norm(center1 - center2)
        dis_pix_ratio = stand_dis / distance
        print("蓝色圆心间的距离为：", distance)
        print("距离像素比--dis_pix_ratio", dis_pix_ratio)

    # 定义红色区域的HSV范围

    # lower_red = lower
    # upper_red = upper
    lower_red = np.array([42, 103, 230])
    upper_red = np.array([78, 198, 255])
    mask = cv2.inRange(hsv, lower_red, upper_red)


    cv2.imshow('Mask', mask)
    cv2.waitKey(100)


    # 提取红色轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours_red = choose_contours(contours)
    draw_contour(filtered_contours_red)
    # print(len(filtered_contours_red))


    # 计算红色圆心间像素距离和实际距离
    if len(contours) >= 2:
        print(len(contours))
        # cv2.contourArea(contour)
        (x1, y1), radius = cv2.minEnclosingCircle(contours[0])
        center1 = [int(x1), int(y1)]
        # center1 = np.array(contours[0][0][0])
        (x2, y2), radius = cv2.minEnclosingCircle(contours[1])
        center2 = [int(x2), int(y2)]
        # center2 = np.array(contours[1][0][0])
        cv2.circle(image, center1, 2, (0, 0, 255), -1)
        cv2.circle(image, center2, 2, (0, 0, 255), -1)
        result = [x1 - x2, y1 - y2]
        distance = np.linalg.norm(result)
        dis_meas.append(distance*dis_pix_ratio)
        print("红色圆心间的像素距离：", distance)
        print("红色圆心间的实际距离：", distance * dis_pix_ratio)
        print("----------------------------")
        # 显示帧
        cv2.imshow('Frame', image)
        
        # 按下'q'键退出
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break

    # 释放视频和关闭窗口
cap.release()
cv2.destroyAllWindows()

