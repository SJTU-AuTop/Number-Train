'''
可视化生成好的测试数据和训练数据
主要用于DEBUG，检查数据集生成的代码是否有问题
'''
from tqdm import tqdm
import os
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    os.makedirs("im2show", exist_ok=True)

    xs = np.load("xs.npy")
    ys = np.load("ys.npy")
    # xs = np.load("test_x.npy")
    # ys = np.load("test_y.npy")
    for i, (img, label) in tqdm(enumerate(zip(xs, ys)), total=xs.shape[0]):
        plt.imshow(img)
        plt.title(f"{label}")

        # 此处切换直接显示图片还是保存图片
        # 主要用于在服务器上训练时，没法直接显示图片
        # plt.savefig(f"im2show/{i}.jpg")
        plt.show()
