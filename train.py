# %%
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from argparse import Namespace
from model.gbcnn import GBCNN
import tensorflow as tf
from keras.utils import np_utils
from Libs.config import get_config

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()


def prepare_data(X, size, channels):
    X = X.reshape(X.shape[0], size, size, channels)
    X = X.astype("float32")
    return X/255.


X_train = prepare_data(x_train, 32, 3)
X_test = prepare_data(x_test, 32, 3)

Y_train = np_utils.to_categorical(y_train, 100)
Y_test = np_utils.to_categorical(y_test, 100)

model = GBCNN(config=get_config())

params = {'config': Namespace(seed=111,
                              boosting_epoch=200,
                              boosting_eta=1e-3,
                              boosting_patience=4,
                              out_dir='checkpoints',
                              additive_epoch=200,
                              additive_batch=128,
                              additive_units=20,
                              additive_eta=1e-3,
                              additive_patience=200)}

model.set_params(**params)
print(model.get_params())
model.fit(X_train, Y_train)
print(f"GB-CNN SCORE:{model.score(X_test, Y_test)}")
pred_stage = [pred for pred in model.predict_stage(X_test)]
yy = [np.argmax(yy, axis=None, out=None) for yy in Y_test]
acc = [accuracy_score(yy, pred) for pred in pred_stage]
plt.plot(acc)
