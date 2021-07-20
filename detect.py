'''
使用电脑的USB摄像头测试模型识别效果
'''
import numpy as np
import time
from tensorflow.keras.models import load_model, Model
import cv2


def detect():
    # 加载训练好的模型。根据自己情况进行修改
    m = load_model("models/model_49_0.8570.h5")

    cap = cv2.VideoCapture(0)
    ok = True
    k = 0
    while ok and k != ord('q'):
        ok, img = cap.read()
        # USB相机默认分辨率为640*480，从中间crop成640*256。
        img = img[112:368]
        # 将640*256分辨率resize成160*64。并进行BGR to RGB以及归一化
        x = cv2.resize(img, (160, 64))[np.newaxis, ..., (2, 1, 0)] / 128. - 1
        # 取置信度最大的一项作为分类结果
        y = np.argmax(m.predict(x)[0])
        cv2.putText(img, str(y), (30, 30), 1, 1, (0, 255, 0), 2)
        cv2.imshow("result", img)
        k = cv2.waitKey(10)


if __name__ == "__main__":
    detect()
