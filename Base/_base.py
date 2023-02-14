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

    def _layer_freezing(self, model):
        model.get_layer(model.layers[-2].name).trainable = False
        assert model.get_layer(model.layers[-2].name).trainable == False, self.log_sh.error(
            "The intermediate dense layer is not frozen!")

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

    def _add(self, model, step):
        self._models.append(model)
        self.steps.append(step)

    def _lists_initialization(self):
        self.g_history = {"loss_train": [],
                          "acc_train": [],
                          "acc_val": []}
        self._models = []
        self.steps = []
        self.layers = []

    def _early_stop(self):
        es = tf.keras.callbacks.EarlyStopping(monitor="mean_squared_error",
                                              patience=self.config.patience,
                                              verbose=0)
        return es

    @abstractmethod
    def fit(self):
        """Abstract method for fit"""

    def _in_train_score(self, X, y):
        pred = np.argmax(self._loss.raw_predictions_to_probs(
            self.decision_function(X)), axis=1)
        y = [np.argmax(yy, axis=None, out=None) for yy in y]
        return np.mean(y == pred)

    def _check_params(self):
        """Check validity of parameters."""

        assert self.config.boosting_epoch >= self.config.units, format(
            f"Boosting number {self.config.boosting_epoch} should be greater than the units {self.config.units}.")

        assert self.config.additive_epoch >= 70, format(
            f"The acceptable additive epoch for the model is greater than {70} but received {self.config.additive_epoch}")

        assert self.config.patience <= self.config.additive_epoch, format(
            f"patience should be equal to or less than {self.config.additive_epoch}, but received {self.config.patience}")

        self._validation_data = False
        if self.config.save_records:
            self._validation_data = True
            # Path for saving the checkpoints
            try:
                os.mkdir('records')
            except:
                self.log_sh.warning('dir already exists for record')
                self.log_sh.warning('previous records have been deleted')
                for dirname, _, filenames in os.walk('records'):
                    for record in filenames:
                        os.remove(os.path.join(dirname, record))

        # Clear the GPU memory
        tf.keras.backend.clear_session()

        # Set the seed for Numpy and tf
        tf.random.set_seed(self.config.seed)
        np.random.RandomState(self.config.seed)
        np.set_printoptions(precision=7, suppress=True)

    def decision_function(self, X):

        pred = self._models[0].predict(X)
        raw_predictions = pred * self.steps[0] + self.intercept
        self._pred = raw_predictions

        for model, step in zip(self._models[1:], self.steps[1:]):
            raw_predictions += model.predict(X) * step

        return raw_predictions

    @abstractmethod
    def _validate_y(self, y):
        """validate y and specify the loss function"""

    @abstractmethod
    def predict(self, X):
        """Return the predicted value"""

    @abstractmethod
    def predict_stage(self, X):
        """Return the predicted value of each boosting iteration"""

    @abstractmethod
    def score(self, X, y):
        """Return the score of the GB-CNN model"""


class BaseGBCNN(BaseEstimator):
    def __init__(self, config):
        super().__init__(config)

    def _data_generator(self, X, y):
        data_generator = ImageDataGenerator(rescale=1)
        train_generator = data_generator.flow(
            X, y, batch_size=self.config.batch)
        return train_generator

    def fit(self, X, y, x_test=None, y_test=None):

        y = self._validate_y(y)
        self._check_params()

        # _loss = multi_class_loss()
        self.intercept = self._loss.model0(y)

        self._lists_initialization()
        model = _layers(X=X, y=y)
        acum = np.ones_like(y) * self.intercept

        es = self._early_stop()
        opt = self._optimizer(eta=self.config.additive_eta,
                              decay=False)

        self.log_fh.info("Training Dense Layers with Gradient Boosting")

        T = int(self.config.boosting_epoch/self.config.units)

        for epoch in range(T):

            self.log_fh.info(f"Epoch: {epoch+1} out of {T}")
            residuals = self._loss.derive(y, acum)

            if epoch == 0:
                model.pop()
                model.add(tf.keras.layers.BatchNormalization())
                model.add(tf.keras.layers.Dense(
                    self.config.units, activation="relu"))
                model.add(tf.keras.layers.Dense(y.shape[1]))
            else:
                output_weights = model.layers[-1].get_weights()
                # Remove the final output layer
                model.pop()

                name = model.layers[-1].name[:-1] + str(int(
                    model.layers[-1].name[-1]) + 1) if "name" not in globals() else name[:-1] + str(int(name[:-1])+1)

                # Add a dense layer before the final output layer
                model.add(tf.keras.layers.Dense(
                    self.config.units, activation="relu",  name=name))
                # Add the final output layer back with the correct shape
                output_layer = tf.keras.layers.Dense(
                    units=y.shape[1], name='output')
                model.add(output_layer)
                # Load the stored weights into the output layer
                output_layer.set_weights(output_weights)

            model.compile(loss="mean_squared_error",
                          optimizer=opt,
                          metrics=[tf.keras.metrics.MeanSquaredError()])

            self.log_sh.info(model.summary())

            val_data = (x_test, y_test) if self._validation_data else (X, y)
            self.history = model.fit(x=self._data_generator(X, residuals),
                                     epochs=self.config.additive_epoch,
                                     use_multiprocessing=False,
                                     verbose=1,
                                     validation_data=val_data,
                                     callbacks=[es]
                                     )

            self._layer_freezing(model=model)

            pred = model.predict(X)
            rho = self.config.boosting_eta * \
                self._loss.newton_step(y, residuals, pred)
            acum = acum + rho * pred
            self._add(model, rho)

            self.log_fh.info("       Additive model-MSE: {0:.7f}".format(model.evaluate(X,
                                                                                        residuals,
                                                                                        verbose=0)[1]))

            # loss_mean = np.mean(self._loss(y, acum))
            loss_mean = np.mean(tf.keras.metrics.categorical_crossentropy(
                y, acum, from_logits=False, label_smoothing=0.0, axis=-1))
            self.log_fh.info(
                "       Gradient Loss: {0:.5f}".format(loss_mean))

            if self.config.save_records:
                self.g_history["acc_val"].append(
                    self._in_train_score(x_test, y_test))
                self.g_history["acc_train"].append(self._in_train_score(X, y))
                self.g_history["loss_train"].append(loss_mean)
                self._save_records(epoch)

    def _save_records(self, epoch):
        """save the checking points."""
        def _path(archive):
            # path = os.path.join("records", archive)
            path = os.path.join(os.getcwd(), archive)
            return path

        archives = [('gb_cnn_loss.txt', self.g_history["loss_train"]),
                    ('gb_cnn_intrain_acc_train.txt',
                     self.g_history["acc_train"]),
                    ('gb_cnn_intrain_acc_val.txt', self.g_history["acc_val"]),
                    ('epoch_' + str(epoch) + '_additive_training_loss.txt',
                     self.history.history['loss']),
                    ('epoch_' + str(epoch) + '_additive_training_val_loss.txt', self.history.history['val_loss'])]

        for archive in archives:
            np.savetxt(_path(archive[0]), archive[1])

        self.log_fh.warning(f"Checkpoints are saved")


class BaseDeepGBNN(BaseEstimator):

    def __init__(self, config):
        super().__init__(config)

    def _regressor(self, X, name):
        """Building the additive deep
        regressor of the gradient boosting"""

        model = tf.keras.models.Sequential(name=name)

        # Normalizing the input
        model.add(tf.keras.layers.Normalization(axis=-1))

        # Build the Input Layer
        # Input and flatten layers for Images dataset
        if len(X.shape) > 3:
            model.add(tf.keras.layers.Dense(self.config.units,
                                            input_shape=X.shape[1:],
                                            activation="relu"))
            flatten_layer = tf.keras.layers.Flatten()

        # Input layer for tabular dataset
        else:
            model.add(tf.keras.layers.Dense(self.config.units,
                                            input_dim=X.shape[1],
                                            activation="relu"))
        # Hidden Layers
        # Empowering the network with frozen trained layers
        for layer in self.layers:
            # Importing frozen layers as the intermediate layers of the network
            model.add(layer)

        # Adds one new raw hidden layer with randomized weight
        # get_weights()[0].shape == (self.config.units, self.config.units)
        # get_weights()[1].shape == (self.config.units)
        layer = tf.keras.layers.Dense(self.config.units,
                                      activation="relu")
        layer.trainable = True
        model.add(layer)

        # Adding dropout layer after the last hidden layer
        # model.add(keras.layers.Dropout(rate=0.1))

        try:
            model.add(flatten_layer)
        except:
            pass
        # Output layers
        model.add(tf.keras.layers.Dense(self.n_classes))

        assert model.trainable == True, "Check the model trainability"
        assert model.layers[-2].trainable == True, "The new hidden layer should be trainable."

        return model

    def fit(self, X, y, x_test=None, y_test=None):

        X = X.astype(np.float32)

        y = self._validate_y(y)
        self._check_params()
        self._lists_initialization()

        self.intercept = self._loss.model0(y)
        acum = np.ones_like(y) * self.intercept

        es = self._early_stop()

        opt = self._optimizer(eta=self.config.additive_eta,
                              decay=False)

        T = int(self.config.boosting_epoch/self.config.units)

        for epoch in range(T):

            residuals = self._loss.derive(y, acum)
            residuals = residuals.astype(np.float32)

            model = self._regressor(X=X,
                                    name=str(epoch)
                                    )

            model.compile(loss="mean_squared_error",
                          optimizer=opt,
                          metrics=[tf.keras.metrics.MeanSquaredError()])

            val_data = (x_test, y_test) if self._validation_data else (X, y)
            model.fit(X, residuals,
                      batch_size=self.config.batch,
                      epochs=self.config.additive_epoch,
                      validation_data = val_data, 
                      verbose=False,
                      callbacks=[es],
                      )

            self._layer_freezing(model=model)

            pred = model.predict(X)
            rho = self.config.boosting_eta * \
                self._loss.newton_step(y, residuals, pred)
            acum = acum + rho * pred

            loss_mean = np.mean(self._loss(y, acum))
            self.g_history["loss_train"].append(loss_mean)
            self._add(model, rho)

            if self.config.save_records:
                self.g_history["acc_val"].append(
                    self._in_train_score(x_test, y_test))
                self.g_history["acc_train"].append(self._in_train_score(X, y))
                self.g_history["loss_train"].append(loss_mean)
                self._save_records(epoch)
