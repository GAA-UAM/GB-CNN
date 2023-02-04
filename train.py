# %%
from argparse import Namespace
from model.gbcnn import GBCNN
import tensorflow as tf
from keras.utils import np_utils
from Libs.config import get_config

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()


def prepare_data(X, size, channels):
    X = X.reshape(X.shape[0], size, size, channels)
    X = X.astype("float32")
    return X/255.


X_train = prepare_data(x_train, 28, 1)
X_test = prepare_data(x_test, 28, 1)

Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)

model = GBCNN(config=get_config())
params = {'config': Namespace(seed=111,
                              cnn_epoch=200,
                              cnn_eta=1e-3,
                              cnn_batch_size=256,
                              cnn_patience=3,
                              boosting_epoch=200,
                              boosting_eta=1e-1,
                              fine_tune_epoch=100,
                              fine_tune_batch=64,
                              fine_tune_units=10,
                              fine_tune_eta=1e-1,
                              fine_tune_patience=7,
                              )
                              }

# params['config'].cnn_epoch = 1
# params['config'].boosting_epoch = 1
# params['config'].fine_tune_units = 1
# params['config'].boosting_epoch = 1
# params['config'].fine_tune_epoch = 1
# model.set_params(**params)
# print(model.get_params())
model.fit(X_train, Y_train)
print(f"MODEL SCORE:{model.score(X_test, y_test)}")