'''
create_test.py: 创建测试集
测试集由OpenArt-mini实拍数据构成
由于实拍数据被裁减成96*64，而模型输入为160*64，故对多余部分补零
'''
import numpy as np
import cv2
import os

xs = []
ys = []
for cls in (2, 5, 10):
    for fn in os.listdir(f"data/test/{cls}"):
        img = cv2.imread(f"data/test/{cls}/{fn}")

        # 测试数据左右补零
        x = np.zeros([64, 160, 3], dtype=np.uint8)
        x[:, -96:] = img[..., (2, 1, 0)]            # OpenCV图片格式BGR，转成模型所需格式RGB
        xs.append(x)
        ys.append(cls)

        # 测试数据左侧补零
        x = np.zeros([64, 160, 3], dtype=np.uint8)  # OpenCV图片格式BGR，转成模型所需格式RGB
        x[:, 32:128] = img[..., (2, 1, 0)]
        xs.append(x)
        ys.append(cls)

        # 测试数据右侧补零
        x = np.zeros([64, 160, 3], dtype=np.uint8)  # OpenCV图片格式BGR，转成模型所需格式RGB
        x[:, :96] = img[..., (2, 1, 0)]
        xs.append(x)
        ys.append(cls)
np.save("test_x", xs)
np.save("test_y", ys)
