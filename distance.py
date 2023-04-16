import cv2
import numpy as np
import matplotlib.pyplot as plt

# 定义基准长度
stand_dis = 50 # mm
dis_pix_ratio = 0

# 用来记录最终数据--每一帧中红色点间的实际长度
dis_meas = []
all_frame_cnt = 0
useful_frame_cnt = 0

# 调试阈值用
# def empty(i):
#     pass

# 过滤面积较小的轮廓
def filtered_contours(contours) :
    # 设置面积阈值
    area_threshold = 100  # 可根据需求调整

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
cap = cv2.VideoCapture('test6.MP4')

# # 调试用代码找到合适的阈值，用来产生控制滑条
# cv2.namedWindow("HSV")
# cv2.resizeWindow("HSV", 640, 300)
# cv2.createTrackbar("HUE Min", "HSV", 10, 179, empty)
# cv2.createTrackbar("SAT Min", "HSV", 0, 255, empty)
# cv2.createTrackbar("VALUE Min", "HSV", 111, 255, empty)
# cv2.createTrackbar("HUE Max", "HSV", 77, 179, empty)
# cv2.createTrackbar("SAT Max", "HSV", 183, 255, empty)
# cv2.createTrackbar("VALUE Max", "HSV", 209, 255, empty)

# # 帧率计数器
# frame_cnt = 0
# N = 10

while True:
    # 读取视频帧
    ret, frame = cap.read()
    all_frame_cnt += 1

    # # 帧计数器
    # frame_cnt += 1
    # if (frame_cnt % N != 1) :
    #     continue

    # 如果没有轮廓 终止
    if not ret:
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

    ############# 处理蓝色点 ####################

    # 定义蓝色的HSV范围-- 可以用上面的注释的代码调参数
    lower_blue = np.array([10, 0, 111])
    upper_blue = np.array([77, 183, 209])

    # 生成蓝色区域的遮罩
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    # 对遮罩进行中值滤波以去噪
    mask_blue = cv2.medianBlur(mask_blue, 5)

    # 在原图上绘制蓝色区域
    cv2.imshow('Mask_blue', mask_blue)
    cv2.waitKey(1)


    # 提取蓝色轮廓 -- 并过滤
    contours, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours_blue = filtered_contours(contours)

    # 计算蓝色圆心间的像素距离和距离像素比
    if len(filtered_contours_blue) == 2:
        # print(len(filtered_contours_blue))
        # 计算圆心和半径
        (x1, y1), radius1 = cv2.minEnclosingCircle(filtered_contours_blue[0])
        center1 = [int(x1), int(y1)]
        (x2, y2), radius2 = cv2.minEnclosingCircle(filtered_contours_blue[1])
        center2 = [int(x2), int(y2)]

        # 绘制轮廓的外接圆--黑色
        cv2.circle(image, center1, int(radius1), (0, 0, 0), 2)
        cv2.circle(image, center2, int(radius2), (0, 0, 0), 2)

        # 绘制圆心 -- 红色
        cv2.circle(image, center1, 5, (0, 0, 255), -1)
        cv2.circle(image, center2, 5, (0, 0, 255), -1)

        # 绘制轮廓 -- 蓝色
        cv2.drawContours(image, contours,-1, (255, 0, 0), 5)

        # 计算距离
        result = [x1 - x2, y1 - y2]
        distance = np.linalg.norm(result)
        dis_pix_ratio = stand_dis / distance
        # print("蓝色圆心间的距离为：", distance)
        # print("距离像素比--dis_pix_ratio", dis_pix_ratio)

    # 如果不是两个点，不更新，用上一次的dis_pix_ratio
        
    ############# 处理红色点 ####################

    # 定义红色区域的HSV范围

    lower_red = np.array([76, 108, 170])
    upper_red = np.array([102, 207, 255])
    # lower_red = lower
    # upper_red = upper
    mask = cv2.inRange(hsv, lower_red, upper_red)


    cv2.imshow('Mask', mask)
    cv2.waitKey(1)


    # 提取红色轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours_red = filtered_contours(contours)


    # 计算红色圆心间像素距离和实际距离
    if len(filtered_contours_red) == 2:
        useful_frame_cnt += 1
        # print(len(filtered_contours_red))
        # 计算圆心和半径
        (x1, y1), radius1 = cv2.minEnclosingCircle(filtered_contours_red[0])
        center1 = [int(x1), int(y1)]
        (x2, y2), radius2 = cv2.minEnclosingCircle(filtered_contours_red[1])
        center2 = [int(x2), int(y2)]

        # 绘制轮廓的外接圆--黑色
        cv2.circle(image, center1, int(radius1), (0, 0, 0), 2)
        cv2.circle(image, center2, int(radius2), (0, 0, 0), 2)

        # 绘制圆心 -- 红色
        cv2.circle(image, center1, 5, (0, 0, 255), -1)
        cv2.circle(image, center2, 5, (0, 0, 255), -1)

        # 绘制轮廓 -- 蓝色
        cv2.drawContours(image, contours,-1, (255, 0, 0), 5)
        # 计算距离
        result = [x1 - x2, y1 - y2]
        distance = np.linalg.norm(result)
        dis_meas.append(distance*dis_pix_ratio)
        if distance*dis_pix_ratio >500 :
            cv2.waitKey(0)
        
        # print("红色圆心间的像素距离：", distance)
        # print("红色圆心间的实际距离：", distance * dis_pix_ratio)
        # print("----------------------------")

        # # 显示帧
        # cv2.imshow('Frame', image)
        
        # 按下'q'键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    # 如果没有捕捉到轮廓或轮廓多余两个，append 0，记为无效帧
    else :
        dis_meas.append(0)

cap.release()
cv2.destroyAllWindows()

print('all frame :', all_frame_cnt)
print('useful frame :', useful_frame_cnt)

# 插值出无效帧
# 找到非零值的索引
nonzero_indices = np.nonzero(dis_meas)[0]

# 循环遍历非零值索引，进行线性插值
for i in range(len(nonzero_indices)-1):
    start_index = nonzero_indices[i]
    end_index = nonzero_indices[i+1]
    start_value = dis_meas[start_index]
    end_value = dis_meas[end_index]

    # 计算插值步长
    step = (end_value - start_value) / (end_index - start_index)

    # 对零值进行插值
    for j in range(start_index+1, end_index):
        dis_meas[j] = start_value + step * (j - start_index)


# 绘制曲线
plt.plot(dis_meas)

# 设置 x 轴和 y 轴的标签
plt.xlabel('frame')
plt.ylabel('length/mm')

# 设置图像标题
plt.title("monocular_length_measurement")

# 显示图像
plt.show()
# 释放视频和关闭窗口



