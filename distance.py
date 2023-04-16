import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import json
import sys

import my_module


# 获取命令行参数
filename = sys.argv[1] # 视频文件名
stand_dis = int(sys.argv[2]) # 基准长度 单位 m
dis_pix_ratio = 0

# 用来记录最终数据--每一帧中红色点间的实际长度
dis_meas = []
all_frame_cnt = 0
useful_frame_cnt = 0

# 载入视频
cap = cv2.VideoCapture(filename)

# 读取调整好的阈值参数
with open('blue_config.json', 'r') as f :
    blue_data = f.read()
blue_config = json.loads(blue_data)

blue_lower = (blue_config["h_min"], blue_config["s_min"], blue_config["v_min"])
blue_upper = (blue_config["h_max"], blue_config["s_max"], blue_config["v_max"])
blue_area_threshold = blue_config["area_threshold"]

with open('red_config.json', 'r') as f :
    red_data = f.read()
red_config = json.loads(red_data)

red_lower = (red_config["h_min"], red_config["s_min"], red_config["v_min"])
red_upper = (red_config["h_max"], red_config["s_max"], red_config["v_max"])
red_area_threshold = red_config["area_threshold"]

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
        print("结束")
        break
    
    # 将帧转换为HSV颜色空间 image 为每一帧的HSV图像
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    fps = cap.get(cv2.CAP_PROP_FPS)


    # 将图片转换为HSV格式
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    ############# 处理蓝色点 ####################

    # 生成蓝色区域的遮罩
    mask_blue = cv2.inRange(hsv, blue_lower, blue_upper)

    # 对遮罩进行中值滤波以去噪
    mask_blue = cv2.medianBlur(mask_blue, 5)


    # 提取蓝色轮廓 -- 并过滤
    contours, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours_blue = my_module.filtered_contours(contours, blue_area_threshold)

    # 计算蓝色圆心间的像素距离和距离像素比
    if len(filtered_contours_blue) == 2:
        # 计算圆心和半径
        (x1, y1), radius1 = cv2.minEnclosingCircle(filtered_contours_blue[0])
        center1 = [int(x1), int(y1)]
        (x2, y2), radius2 = cv2.minEnclosingCircle(filtered_contours_blue[1])
        center2 = [int(x2), int(y2)]

        # 计算距离
        result = [x1 - x2, y1 - y2]
        distance = np.linalg.norm(result)
        dis_pix_ratio = stand_dis / distance
        # print("蓝色圆心间的距离为：", distance)
        # print("距离像素比--dis_pix_ratio", dis_pix_ratio)

    # 如果不是两个点，不更新，用上一次的dis_pix_ratio
        
    ############# 处理红色点 ####################

    # 定义红色区域的HSV范围

    mask = cv2.inRange(hsv, red_lower, red_upper)

    # 提取红色轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours_red = my_module.filtered_contours(contours, red_area_threshold)


    # 计算红色圆心间像素距离和实际距离
    if len(filtered_contours_red) == 2:
        useful_frame_cnt += 1
        # 计算圆心和半径
        (x1, y1), radius1 = cv2.minEnclosingCircle(filtered_contours_red[0])
        center1 = [int(x1), int(y1)]
        (x2, y2), radius2 = cv2.minEnclosingCircle(filtered_contours_red[1])
        center2 = [int(x2), int(y2)]

        # 计算距离
        result = [x1 - x2, y1 - y2]
        distance = np.linalg.norm(result)
        dis_meas.append(distance*dis_pix_ratio)


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
t = np.linspace(0, len(dis_meas), len(dis_meas))
t = t / fps
plt.plot(t, dis_meas)

# 设置 x 轴和 y 轴的标签
plt.xlabel('frame')
plt.ylabel('length/mm')

# 设置图像标题
plt.title("monocular_length_measurement")

# 显示图像
plt.show()
# 释放视频和关闭窗口

# 将数据保存为.mat文件
sio.savemat('dis_meas.mat', {'dis_meas': dis_meas})
sio.savemat('dis_t.mat', {'dis_t': t})

# 在MATLAB中加载.mat文件
# 可以使用load函数加载.mat文件并将数据存储在MATLAB变量中
# 例如，load('data.mat')将加载.mat文件并将数据存储在名为data的MATLAB变量中


