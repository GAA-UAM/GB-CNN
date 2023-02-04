""" Base Class of Gradient Boosted - Convolutional Neural Network """

# Author: Seyedsaman Emami
# Author: Gonzalo Martínez-Muñoz

# Licence: GNU Lesser General Public License v2.1 (LGPL-2.1)

import os
import glob
import keras
import numpy as np
import tensorflow as tf
from ._params import Params
from ._losses import multi_class_loss
from Libs._logging import FileHandler, StreamHandler

from abc import abstractmethod
from sklearn.utils.multiclass import check_classification_targets


class BaseEstimator(Params):

    def __init__(self, config):

        self.config = config
        self.log_fh = FileHandler()
        self.log_sh = StreamHandler()

        logs = glob.glob('*log')
        if os.path.getsize(logs[0]) > 0:
            f = open(logs[0], "r+")
            f.truncate()
            self.log_fh.warning(
                "The previously saved log file has been deleted!")

    def _validate_y(self, y):
        check_classification_targets(y)
        y = y.astype('int32')
        return y

    def _early_stopping(self, monitor, patience):
        es = keras.callbacks.EarlyStopping(monitor=monitor,
                                           patience=patience,
                                           verbose=0)
        return es

    def _optimizer(self):
        opt = tf.keras.optimizers.Adam(learning_rate=self.config.learning_rate,
                                       beta_1=0.9,
                                       beta_2=0.999,
                                       epsilon=1e-07,
                                       amsgrad=False,
                                       name="Adam")
        return opt

    def _cnn(self, X, y):

        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Conv2D(32, (3, 3), input_shape=X.shape[1:],
                                         kernel_initializer='he_uniform', padding='same', activation='relu'))
        model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu',
                                         kernel_initializer='he_uniform', padding='same'))
        model.add(tf.keras.layers.MaxPooling2D((2, 2)))
        model.add(tf.keras.layers.Dropout(0.1))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(
            128, activation='relu', kernel_initializer='he_uniform'))
        model.add(tf.keras.layers.Dense(y.shape[1], activation='softmax'))

        model.compile(optimizer=self._optimizer(), loss='categorical_crossentropy',
                      metrics=['accuracy'])

        self.log_fh.info("CNN training")
        self.log_sh.info(model.summary())

        model.fit(X, y,
                  batch_size=self.config.batch_size,
                  epochs=self.config.epoch,
                  verbose=1,
                  callbacks=[self._early_stopping(monitor='loss',
                                                  patience=self.config.patience)])

        model.save("CNN.h5")
        self.log_fh.warning("Pre-Trained CNN is saved (.h5)")
        self.log_fh.info("CNN acc on original data: {0:.3f}".format(model.evaluate(X, y, verbose=0)[1]))
        return model

    def fit(self, X, y):

        _loss = multi_class_loss()

        y = self._validate_y(y)
        self._check_params()
        self._lists_initialization()

        T = int(self.config.boosting_epoch/self.config.unit)

        model = self._cnn(X, y)
        for layer in model.layers:
            layer.trainable = False

        model.compile(loss="mean_squared_error", optimizer="adam")

        self.intercept = _loss.model0(y)
        acum = np.ones_like(y) * self.intercept

        self.log_fh.info("Training Dense Layers with Gradient Boosting")

        for epoch in range(T):

            self.log_fh.info(f"Epoch: {epoch} out of {T-1}")

            residuals = _loss.derive(y, acum)

            # Get the output of the pre-trained model
            out = model.layers[-2].output
            # Add new dense layers
            out = tf.keras.layers.Dense(
                units=self.config.unit, activation="relu")(out)
            # Add the final layer for prediction
            out = tf.keras.layers.Dense(units=y.shape[1])(out)

            # Create a new model with the dense layers on top of the pre-trained model
            model = tf.keras.models.Model(inputs=model.input, outputs=out)

            model.compile(loss="mean_squared_error",
                          optimizer=self._optimizer(),
                          metrics=[tf.keras.metrics.MeanSquaredError()])

            self.log_sh.info(model.summary())

            model.fit(X, residuals,
                      batch_size=200,
                      epochs=self.config.additive_epoch,
                      verbose=1,
                      callbacks=[self._early_stopping(monitor="mean_squared_error",
                                                      patience=self.config.patience)]
                      )

            self._layer_freezing(model=model)

            pred = model.predict(X)
            rho = self.config.eta * _loss.newton_step(y, residuals, pred)
            acum = acum + rho * pred
            self._add(model, rho)

            self.log_fh.info("      Additive model-MSE: {0:.7f}".format(model.evaluate(X,
                                                                                 residuals,
                                                                                 verbose=0)[1]))
            loss_curve = np.mean(_loss(y, acum))
            self.log_fh.info(
                "       Gradient Loss: {0:.5f}".format(loss_curve))
            self._loss_curve.append(loss_curve)


    def _layer_freezing(self, model):
        name = model.layers[-2].name
        model.get_layer(name).trainable = False
        assert model.get_layer(
            name).trainable == False, self.log_sh.error("The intermediate dense layer is not frozen!")

    def _add(self, model, step):
        self._models.append(model)
        self.steps.append(step)

    def decision_function(self, X):

        pred = self._models[0].predict(X)
        raw_predictions = pred * self.steps[0] + self.intercept
        self._pred = raw_predictions

        for model, step in zip(self._models[1:], self.steps[1:]):
            raw_predictions += model.predict(X) * step

        return raw_predictions

    def _check_params(self):
        """Check validity of parameters."""

        if self.config.boosting_epoch < self.config.unit:
            raise ValueError(
                f"Boosting number {self.config.boosting_epoch} should be greater than the units {self.config.unit}.")

        tf.random.set_seed(self.config.seed)
        np.random.RandomState(self.config.seed)

    def _lists_initialization(self):
        self._loss_curve = []
        self._models = []
        self.steps = []

    @abstractmethod
    def predict(self, X):
        """Return the predicted value"""

    @abstractmethod
    def predict_stage(self, X):
        """Return the predicted value of each boosting iteration"""

    @abstractmethod
    def score(self, X, y):
        """Return the score (accuracy for classification and aRMSE for regression)"""

    @classmethod
    def _clear_session(cls):
        """Clear the global state from stored models"""
        tf.keras.backend.clear_session()
