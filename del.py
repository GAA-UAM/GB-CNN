# %%
import tensorflow as tf
from keras.utils import np_utils

model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(32, (3, 3), input_shape=(28, 28, 1),
          kernel_initializer='he_uniform', padding='same', activation='relu'))
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu',
          kernel_initializer='he_uniform', padding='same'))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Dropout(0.1))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(
    128, activation='relu', kernel_initializer='he_uniform'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))


opt = tf.keras.optimizers.Adam(learning_rate=0.1,
                               beta_1=0.9,
                               beta_2=0.999,
                               epsilon=1e-07,
                               amsgrad=False,
                               name="Adam")

model.compile(optimizer=opt, loss='categorical_crossentropy',
              metrics=['accuracy'])


model.summary()
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()


def prepare_data(X, size, channels):
    X = X.reshape(X.shape[0], size, size, channels)
    X = X.astype("float32")
    return X/255.


X_train = prepare_data(x_train, 28, 1)
X_test = prepare_data(x_test, 28, 1)

Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)


model.fit(X_train, Y_train,
          batch_size=128,
          epochs=10,
          verbose=0,
          validation_data=(X_test, Y_test))

# %%
