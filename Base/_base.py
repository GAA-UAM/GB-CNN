""" Base Class of Gradient Boosted - Convolutional Neural Network """

# Author: Seyedsaman Emami
# Author: Gonzalo Martínez-Muñoz

# Licence: GNU Lesser General Public License v2.1 (LGPL-2.1)

import os
import glob
import numpy as np
import tensorflow as tf
from ._params import Params
from ._losses import multi_class_loss
from Libs._logging import FileHandler, StreamHandler
from tensorflow.keras.preprocessing.image import ImageDataGenerator


from ._cnn import _layers
from abc import abstractmethod


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

    def _cnn_es(self, monitor, patience):
        es = tf.keras.callbacks.EarlyStopping(monitor=monitor,
                                              patience=patience,
                                              verbose=0)
        return es

    def _optimizer(self, eta=1e-3, decay=False):

        if decay:
            initial_learning_rate = eta
            eta = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate,
                decay_steps=500,
                decay_rate=0.96,
                staircase=True)

        opt = tf.keras.optimizers.Adam(
            learning_rate=eta,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07,
            amsgrad=False,
            name="Adam")

        return opt

    def _boosting_es(self, loss, min_loss, patience):
        counter = 0
        if loss > min_loss:
            counter += 1
            if counter >= patience:
                return True

    def _data_generator(self, X, y):
        data_generator = ImageDataGenerator(rescale=1)
        train_generator = data_generator.flow(
            X, y, batch_size=self.config.additive_batch)
        return train_generator

    def _layer_freezing(self, model):
        name = model.layers[-2].name
        model.get_layer(name).trainable = False
        assert model.get_layer(
            name).trainable == False, self.log_sh.error("The intermediate dense layer is not frozen!")

    def _add(self, model, step):
        self._models.append(model)
        self.steps.append(step)

    def _lists_initialization(self):
        self._loss_curve = []
        self._models = []
        self.steps = []

    def fit(self, X, y):

        _loss = multi_class_loss()

        y = self._validate_y(y)
        self._check_params()
        self._lists_initialization()

        T = int(self.config.boosting_epoch/self.config.additive_units)

        model = _layers(X=X, y=y)

        self.intercept = _loss.model0(y)
        acum = np.ones_like(y) * self.intercept

        self.log_fh.info("Training Dense Layers with Gradient Boosting")

        for epoch in range(T):

            self.log_fh.info(f"Epoch: {epoch+1} out of {T}")

            residuals = _loss.derive(y, acum)

            if epoch == 0:
                model.pop()
                model.add(tf.keras.layers.BatchNormalization())
                model.add(tf.keras.layers.Dense(
                    self.config.additive_units, activation="relu"))
                model.add(tf.keras.layers.Dense(y.shape[1]))
            else:
                output_weights = model.layers[-1].get_weights()
                # Remove the final output layer
                model.pop()
                # Add a dense layer before the final output layer
                model.add(tf.keras.layers.Dense(
                    self.config.additive_units, activation="relu"))
                # Add the final output layer back with the correct shape
                output_layer = tf.keras.layers.Dense(units=y.shape[1])
                model.add(output_layer)
                # Load the stored weights into the output layer
                output_layer.set_weights(output_weights)

            model.compile(loss="mean_squared_error",
                          optimizer=self._optimizer(
                              eta=self.config.additive_eta, decay=False),
                          metrics=[tf.keras.metrics.MeanSquaredError()])

            self.log_sh.info(model.summary())

            self.history = model.fit(x=self._data_generator(X, residuals),
                                     #  batch_size=self.config.additive_batch,
                                     epochs=self.config.additive_epoch,
                                     use_multiprocessing=False,
                                     verbose=1,
                                     callbacks=[self._cnn_es(monitor="mean_squared_error",
                                                             patience=self.config.additive_patience)]
                                     )

            self._layer_freezing(model=model)

            pred = model.predict(X)
            rho = self.config.boosting_eta * \
                _loss.newton_step(y, residuals, pred)
            acum = acum + rho * pred
            self._add(model, rho)

            self.log_fh.info("       Additive model-MSE: {0:.7f}".format(model.evaluate(X,
                                                                                        residuals,
                                                                                        verbose=0)[1]))
            loss_mean = np.mean(_loss(y, acum))
            self.log_fh.info(
                "       Gradient Loss: {0:.5f}".format(loss_mean))
            self._loss_curve.append(loss_mean)
            self._save_checkpoint(self._models[-1], epoch)

            if self._boosting_es(loss_mean, np.min(self._loss_curve), self.config.boosting_patience):
                self.log_fh.warning(
                    "Boosting training is stopped (Early stopping)")
                break

    def decision_function(self, X):

        pred = self._models[0].predict(X)
        raw_predictions = pred * self.steps[0] + self.intercept
        self._pred = raw_predictions

        for model, step in zip(self._models[1:], self.steps[1:]):
            raw_predictions += model.predict(X) * step

        return raw_predictions

    def _check_params(self):
        """Check validity of parameters."""

        assert self.config.boosting_epoch >= self.config.additive_units, format(
            f"Boosting number {self.config.boosting_epoch} should be greater than the units {self.config.additive_units}.")

        if not self.config.out_dir:
            os.mkdir('checkpoints')

        # clear the GPU memory
        tf.keras.backend.clear_session()

        tf.random.set_seed(self.config.seed)
        np.random.RandomState(self.config.seed)
        np.set_printoptions(precision=7, suppress=True)

    def _validate_y(self, y):
        assert len(y.shape) == 2, "input shape is not valid!"
        y = y.astype('int32')
        return y

    def additive_history(self):
        training_loss = self.history.history['loss']
        training_mse = self.history.history['mean_squared_error']
        return training_loss

    def _save_checkpoint(self, model, epoch):
        """Dump the trained GB_CNN model."""
        model_name = os.path.join(
            self.config.out_dir, f'{model.name + str(epoch)}.h5')
        self.log_fh.warning(f"Checkpoint {model.name} is saved")
        model.save(model_name)

    def score_cnn(self, X, y):
        assert len(y.shape) == 2, "input shape is not valid"
        model = tf.keras.models.load_model(glob.glob("*h5")[0])
        score = model.evaluate(X, y, verbose=0)
        return score

    @abstractmethod
    def predict(self, X):
        """Return the predicted value"""

    @abstractmethod
    def predict_stage(self, X):
        """Return the predicted value of each boosting iteration"""

    @abstractmethod
    def score(self, X, y):
        """Return the score of the GB-CNN model"""

    @classmethod
    def _clear_session(cls):
        """Clear the global state from stored models"""
        tf.keras.backend.clear_session()
