'''
使用Keras进行模型的训练
需要事先准备好训练数据和测试数据
由于该代码使用时是在服务器上训练，如果在PC上运行可能报错内存不足，此时需要降低batch_size或者减少训练数据大小
'''
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from model import create_model160x64
from dataset import MyDataset
import numpy as np
import os

xs = np.load('xs.npy')
ys = np.load('ys.npy')

x_train = xs
y_train = ys

x_train = x_train / 128.0 - 1
y_train = to_categorical(y_train)

x_test = np.load("test_x.npy")
y_test = np.load("test_y.npy")

x_test = x_test / 128. - 1.
y_test = to_categorical(y_test)

# xs = np.concatenate([x_train, x_test], axis=0)
# ys = np.concatenate([y_train, y_test], axis=0)
# x_train, x_test, y_train, y_test = train_test_split(xs, ys, test_size=0.1)

if __name__ == "__main__":
    if not (os.path.exists('models')):
        os.mkdir("models")
    model = create_model160x64(width=0.5)
    model.summary()

    opt = Adam(lr=0.002)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=["acc"])
    early_stop = EarlyStopping(patience=20)
    reduce_lr = ReduceLROnPlateau(patience=15)
    save_weights = ModelCheckpoint("models/model_{epoch:02d}_{val_acc:.4f}.h5",
                                   save_best_only=True, monitor='val_acc')
    callbacks = [save_weights, reduce_lr, early_stop]
    model.fit(x_train, y_train, epochs=100, batch_size=64,
              validation_data=(x_test, y_test), callbacks=callbacks)
