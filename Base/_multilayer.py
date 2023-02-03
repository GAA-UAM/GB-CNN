""" Deep - Gradient Boosted Neural Network """

# Author: Seyedsaman Emami
# Author: Gonzalo Martínez-Muñoz

# Licence: GNU Lesser General Public License v2.1 (LGPL-2.1)

import keras
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from ._base import BaseEstimator
from ._losses import multi_class_loss
from abc import abstractmethod


class BaseMultilayer(BaseEstimator):

    def __init__(self,
                 iter=50,
                 eta=0.1,
                 learning_rate=1e-3,
                 total_nn=10,
                 num_nn_step=1,
                 batch_size=128,
                 early_stopping=10,
                 verbose=False,
                 random_state=None
                 ):

        self.iter = iter
        self.eta = eta
        self.learning_rate = learning_rate
        self.total_nn = total_nn
        self.num_nn_step = num_nn_step
        self.batch_size = batch_size
        self.early_stopping = early_stopping
        self.verbose = verbose
        self.random_state = random_state

    @abstractmethod
    def _validate_y(self, y):
        """validate y and specify the loss function"""

    def _early_stopping(self, monitor, patience):
        es = keras.callbacks.EarlyStopping(monitor=monitor,
                                           patience=patience,
                                           verbose=0)
        return es

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
        model.add(tf.keras.layers.Dense(10, activation='softmax'))

        opt = tf.keras.optimizers.Adam(learning_rate=0.1,
                                       beta_1=0.9,
                                       beta_2=0.999,
                                       epsilon=1e-07,
                                       amsgrad=False,
                                       name="Adam")

        model.compile(optimizer=opt, loss='categorical_crossentropy',
                      metrics=['accuracy'])

        model.fit(X, y,
                  batch_size=200,
                  epochs=2,
                  verbose=1,
                  callbacks=[self._early_stopping(monitor='loss', patience=5)])

        model.save("CNNs.h5")

        return model

    def _load_model(self, X, y):
        model = self._cnn(X, y)
        for layer in model.layers:
            layer.trainable = False
        return model

    def fit(self, X, y):

        _loss = multi_class_loss()

        y = self._validate_y(y)
        self._check_params()
        self._lists_initialization()

        T = int(self.total_nn/self.num_nn_step)
        epochs = self.iter

        model = self._load_model(X, y)

        model.compile(loss="mean_squared_error", optimizer="adam")

        self.intercept = _loss.model0(y)
        acum = np.ones_like(y) * self.intercept

        patience = self.iter if not self.early_stopping else self.early_stopping
        es = keras.callbacks.EarlyStopping(monitor="",
                                           patience=patience,
                                           verbose=0)

        opt = tf.keras.optimizers.Adam(learning_rate=self.learning_rate,
                                       beta_1=0.9,
                                       beta_2=0.999,
                                       epsilon=1e-07,
                                       amsgrad=False,
                                       name="Adam")

        for i in tqdm(range(T)):

            residuals = _loss.derive(y, acum)

            # Get the output of the pre-trained model
            out = model.layers[-2].output
            # Add new dense layers
            out = tf.keras.layers.Dense(units=64, activation="relu")(out)

            # Add the final layer for prediction
            out = tf.keras.layers.Dense(units=10)(out)

            # Create a new model with the dense layers on top of the pre-trained model
            model = tf.keras.models.Model(inputs=model.input, outputs=out)

            model.compile(loss="mean_squared_error",
                          optimizer=opt,
                          metrics=[tf.keras.metrics.MeanSquaredError()])

            model.fit(X, residuals,
                      batch_size=200,
                      epochs=10,
                      verbose=self.verbose,
                      callbacks=[self._early_stopping(
                          monitor="mean_squared_error", patience=2)],
                      )

            self._layer_freezing(model=model)

            pred = model.predict(X)
            rho = self.eta * _loss.newton_step(y, residuals, pred)
            acum = acum + rho * pred

            self._reg_score.append(model.evaluate(X,
                                                  residuals,
                                                  verbose=0)[1])
            self._loss_curve.append(np.mean(_loss(y, acum)))
            self._add(model, rho)

    def _layer_freezing(self, model):
        name = model.layers[-2].name
        model.get_layer(name).trainable = False
        assert model.get_layer(
            name).trainable == False, "The intermediate dense layer is not frozen!"

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

        if self.total_nn < self.num_nn_step:
            raise ValueError(
                f"Boosting number {self.total_nn} should be greater than the units {self.num_nn_step}.")

        if self.random_state is None:
            raise ValueError("Expected `seed` argument to be an integer")
        else:
            tf.random.set_seed(self.random_state)
            np.random.RandomState(self.random_state)

    def _lists_initialization(self):
        self._reg_score = []
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
