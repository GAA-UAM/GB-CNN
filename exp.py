#%%
from model.gbcnn import GBCNN
import tensorflow as tf
from keras.utils import np_utils
from model.config import get_config

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
model.fit(X_train, Y_train)
# %%
# import numpy as np

# pred= model.predict(X_test)
# #%%
# np.unique(pred)

# #%%
# model.score(X_test, Y_test)
# # %%
# mo = tf.keras.models.load_model("CNNs.h5")
# mo.evaluate(X_test,
#                                Y_test,
#                                verbose=0)