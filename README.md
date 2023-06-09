# 用途
- 拉伸实验，测量软体材料的伸长量，代替引伸计作用

# 标定
- 实验前在白纸上画出两个蓝色点，圆心间距为50mm（或者为l，自行调整程序中stand_dis值）作为基准

# 原理
- 两蓝色点的实际长度为l_b, 像素长度为l_b_pix
- 计算得到距离像素比--dis_pix_ratio，用来换算实际长度
- 用两个红色点像素长度l_r_pix
- 两个红色点的实际长度是l_r_pix*dis_pix_ratio

# 像素距离的计算
- 将每一帧图片转为HSV
- 对于蓝色和红色圆点采用相应的lower和upper进行inRange（遮罩），通过调整合适的HSV值，可以使得图片中只含有目标圆点
- 提取轮廓
- 过滤轮廓，略去面积较小的轮廓
- 根据轮廓计算外接圆圆心和半径
- 通过圆心距离来计算像素距离
- 对于没有捕捉到轮廓的点，记录为0，后面采用插值的方式补充


# 阈值调节方法
将37-45行和68-77行，注释开启，并将红色或蓝色的阈值进行替换
~~~py
# 调节蓝色
lower_blue = lower 
upper_blue = upper
# 调节红色
lower_red = lower 
upper_red = upper
~~~

# 使用方法
## 先调用config_adj.py 
使用下面命令，调整滑条，找到合适的值
- 调整蓝点参数

  python config_adj.py [视频文件名] blue 
- 调整红点参数

  python config_adj.py [视频文件名] red
## 后调用distance获取长度
python distance.py [视频文件名] [基准长度]
# 修改帧率
- 通过帧率计数器调整帧率
