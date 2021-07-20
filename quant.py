'''
使用tflite进行int8量化，和官方参考代码相差无几
'''
from tensorflow import __version__ as v

need_version = '1.14.0'
if v < need_version:
    raise Exception("Tf's version must bigger than %s" % need_version)
import tensorflow.compat.v2 as tf
from tensorflow.keras.models import load_model
import pathlib
import argparse
import numpy as np
import os

from model import export_model

tf.enable_v2_behavior()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-model', help='model to be converted', default='./models/model_47_0.9625.h5', type=str)
    parser.add_argument('-d', '--data', help='represent data', type=str, default=r'xs.npy')
    parser.add_argument('-q', '--quant', help='quant model', action='store_true', default=True)
    parser.add_argument('-n', '--num', help='how many datas as the represent data', type=int, default=10)
    parser.add_argument('--mean', help='mean value', type=int, default=128)
    parser.add_argument('--std', help='std value', type=int, default=128)
    args, unknown = parser.parse_known_args()

    model = args.model
    data_path = args.data
    quant = args.quant
    num = args.num

    mean = args.mean
    std = args.std

    print(model)
    model_name = os.path.splitext(model)[0]
    
    # 在这里不是直接加载训练好的模型，而是加载专门定义好的用于导出的模型
    # 注意这里的模型宽度参数必须和训练时保持一致
    m = export_model(0.5)
    m.load_weights(model)
    # m = load_model(model)

    converter = tf.lite.TFLiteConverter.from_keras_model(m)
    tflite_model = converter.convert()

    tflite_models_dir = pathlib.Path("./")
    tflite_models_dir.mkdir(exist_ok=True, parents=True)

    tflite_name = '%s.tflite' % model_name
    tflite_model_file = tflite_models_dir / tflite_name
    tflite_model_file.write_bytes(tflite_model)

    if quant:
        converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
        converter.target_spec.supportes_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        if data_path:
            data = np.load(data_path)
            images = (tf.cast(data, tf.float32) - mean) / std
            data_ds = tf.data.Dataset.from_tensor_slices((images)).batch(1)


            def representative_data_gen():
                for input_value in data_ds.take(500):
                    yield [input_value]


            converter.representative_dataset = representative_data_gen

        tflite_model_quant = converter.convert()
        tflite_model_quant_dir = pathlib.Path('./')
        m_path = '%s_quant.tflite' % model_name
        tflite_model_quant_file = tflite_model_quant_dir / m_path
        tflite_model_quant_file.write_bytes(tflite_model_quant)
