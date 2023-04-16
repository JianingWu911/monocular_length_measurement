
import cv2
import numpy as np
import json
import sys

import my_module 


# 调试阈值用
def empty(i):
    pass

# 获取命令行参数
filename = sys.argv[1] # 视频文件名
color = sys.argv[2] #处理何种颜色
json_name = color + '_config.json' # 组合成JSON文件名
# 载入视频
cap = cv2.VideoCapture(filename)

# 调试用代码找到合适的阈值，用来产生控制滑条
with open(json_name, 'r') as f :
    data = f.read()
config = json.loads(data)
h_min = config["h_min"]
s_min = config["s_min"]
v_min = config["v_min"]
h_max = config["h_max"]
s_max = config["s_max"]
v_max = config["v_max"]
area_threshold = config["area_threshold"]

print(h_min, s_min, v_min, h_max, s_max, v_max)

cv2.namedWindow("HSV")
cv2.resizeWindow("HSV", 640, 300)
cv2.createTrackbar("HUE Min", "HSV", h_min, 179, empty)
cv2.createTrackbar("SAT Min", "HSV", s_min, 255, empty)
cv2.createTrackbar("VALUE Min", "HSV", v_min, 255, empty)
cv2.createTrackbar("HUE Max", "HSV", h_max, 179, empty)
cv2.createTrackbar("SAT Max", "HSV", s_max, 255, empty)
cv2.createTrackbar("VALUE Max", "HSV", v_max, 255, empty)
cv2.createTrackbar("AREA_THRESHOLD", "HSV", area_threshold, 1000, empty)


while True:
    # 读取视频帧
    ret, frame = cap.read()

    # 如果没有轮廓 终止
    if not ret:
        break
    
    # 将帧转换为HSV颜色空间 image 为每一帧的HSV图像
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 调试用代码找到合适的阈值，用来产生控制滑条
    if cv2.getWindowProperty("HSV", cv2.WND_PROP_VISIBLE) == 0: # 不存在窗口跳出循环
        break
    h_min = cv2.getTrackbarPos("HUE Min", "HSV")
    h_max = cv2.getTrackbarPos("HUE Max", "HSV")
    s_min = cv2.getTrackbarPos("SAT Min", "HSV")
    s_max = cv2.getTrackbarPos("SAT Max", "HSV")
    v_min = cv2.getTrackbarPos("VALUE Min", "HSV")
    v_max = cv2.getTrackbarPos("VALUE Max", "HSV")
    area_threshold = cv2.getTrackbarPos("AREA_THRESHOLD", "HSV")
    
    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])

    # 将图片转换为HSV格式
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    ############# 处理红色点 ####################

    # 生成红色区域的遮罩
    mask = cv2.inRange(hsv, lower, upper)

    # 对遮罩进行中值滤波以去噪
    mask = cv2.medianBlur(mask, 5)

    # 提取红色轮廓 -- 并过滤
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = my_module.filtered_contours(contours, area_threshold)

    if len(filtered_contours) == 2:
        # 计算圆心和半径
        (x1, y1), radius1 = cv2.minEnclosingCircle(filtered_contours[0])
        center1 = [int(x1), int(y1)]
        (x2, y2), radius2 = cv2.minEnclosingCircle(filtered_contours[1])
        center2 = [int(x2), int(y2)]

        # 绘制圆心 -- 红色
        cv2.circle(image, center1, 5, (0, 0, 255), -1)
        cv2.circle(image, center2, 5, (0, 0, 255), -1)

        # 绘制轮廓 -- 蓝色
        cv2.drawContours(image, filtered_contours,-1, (255, 0, 0), 2)
         
        # 绘制轮廓的外接圆--黑色
        cv2.circle(image, center1, int(radius1), (0, 0, 0), 2)
        cv2.circle(image, center2, int(radius2), (0, 0, 0), 2)

    # 在原图上绘制红色区域
    cv2.imshow('Mask', mask)
    cv2.imshow('image', image)
    cv2.waitKey(100)
        

cap.release()
cv2.destroyAllWindows()

data = {
    "h_min": h_min,
    "h_max": h_max,
    "s_min": s_min,
    "s_max": s_max,
    "v_min": v_min,
    "v_max": v_max,
    "area_threshold" : area_threshold
}


with open(json_name, "w") as f:
    json.dump(data, f)  # 将字典对象保存到 JSON 文件中
print(h_min, s_min, v_min, h_max, s_max, v_max)
