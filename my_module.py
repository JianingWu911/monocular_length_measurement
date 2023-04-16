import cv2
# 过滤面积较小的轮廓
def filtered_contours(contours, area_threshold) :

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