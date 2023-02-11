# %%
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from argparse import Namespace
from model.gbcnn import GBCNN
import tensorflow as tf
from keras.utils import np_utils
from Libs.config import get_config

gpu_memory_fraction = 0.7

# Create GPUOptions with the fraction of GPU memory to allocate
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)

# Create a session with the GPUOptions
session = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()


def prepare_data(X, size, channels):
    X = X.reshape(X.shape[0], size, size, channels)
    X = X.astype("float32")
    return X/255.


X_train = prepare_data(x_train, 32, 3)
X_test = prepare_data(x_test, 32, 3)

Y_train = np_utils.to_categorical(y_train, 100)
Y_test = np_utils.to_categorical(y_test, 100)


with tf.device('/gpu:0'):
    model = GBCNN(config=get_config())

    params = {'config': Namespace(seed=111,
                                boosting_epoch=1,
                                boosting_eta=1e-3,
                                boosting_patience=4,
                                save_check_points=True,
                                additive_epoch=1,
                                additive_batch=200,
                                additive_units=1,
                                additive_eta=1e-3,
                                additive_patience=200)}

    model.set_params(**params)
    print(model.get_params())
    model.fit(X_train, Y_train, X_test, Y_test)
    
print(f"GB-CNN SCORE:{model.score(X_test, Y_test)}")
pred_stage = [pred for pred in model.predict_stage(X_test)]
yy = [np.argmax(yy, axis=None, out=None) for yy in Y_test]
acc = [accuracy_score(yy, pred) for pred in pred_stage]
plt.plot(acc)