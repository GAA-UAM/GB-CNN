import tensorflow as tf

def _layers(X, y):

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(32, (3, 3), input_shape=X.shape[1:],
                                        kernel_initializer='he_uniform', padding='same', activation='relu'))
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu',
                                        kernel_initializer='he_uniform', padding='same'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Dropout(0.2))

    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu',
                                        kernel_initializer='he_uniform', padding='same'))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu',
                                        kernel_initializer='he_uniform', padding='same'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu',
                                        kernel_initializer='he_uniform', padding='same'))
    model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu',
                                        kernel_initializer='he_uniform', padding='same'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Dropout(0.4))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(32, activation='relu', kernel_initializer='he_uniform'))
    model.add(tf.keras.layers.Dense(y.shape[1], activation='softmax'))

    return model