""" Deep Gradient Boosted Neural Network """

# Author: Seyedsaman Emami
# Author: Gonzalo Martínez-Muñoz

# Licence: GNU Lesser General Public License v2.1 (LGPL-2.1)

import numpy as np
from Base import _losses
from Base._base import BaseDeepGBNN
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils.multiclass import type_of_target, check_classification_targets


class DeepClassifier(BaseDeepGBNN):

    def __init__(self, config):

        super().__init__(config)

    def _validate_y(self, y):

        check_classification_targets(y)
        if type_of_target(y) == "binary":
            self._loss = _losses.log_exponential_loss()
            self.n_classes = 1
            lb = LabelBinarizer(pos_label=1, neg_label=-1)
            y = lb.fit_transform(y).ravel()
        else:
            self._loss = _losses.multi_class_loss()
            if len(y.shape) < 2:
                lb = LabelBinarizer()
                y = lb.fit_transform(y)
            self.n_classes = y.shape[1]
        y = y.astype('int32')

        return y

    def predict_proba(self, X):
        return self._loss.raw_predictions_to_probs(self.decision_function(X))

    def _predict(self, probs):
        return np.argmax(probs, axis=1)

    def predict(self, X):
        pred = self.predict_proba(X)
        return self._predict(pred)

    def predict_stage(self, X):
        raw_predictions = self._pred
        for model, step in zip(self._models, self.steps):
            raw_predictions += model.predict(X) * step
            probs = self._loss.raw_predictions_to_probs(raw_predictions)
            pred = np.argmax(probs, axis=1)
            yield pred

    def score(self, X, y):
        """Returns average accuracy for all class labels."""
        X = X.astype(np.float32)
        y = y.astype(np.int32)
        score = y == self.predict(X)
        return np.mean(score)


class DeepRegressor(BaseDeepGBNN):

    def __init__(self,
                 additive_epoch=10,
                 additive_eta=1e-3,
                 additive_unit=1,
                 boosting_epoch=200,
                 boosting_eta=0.1,
                 batch_size=128,
                 early_stopping=False,
                 random_state=None):

        super().__init__(additive_epoch,
                         additive_eta,
                         additive_unit,
                         boosting_epoch,
                         boosting_eta,
                         batch_size,
                         early_stopping,
                         random_state)

    def _validate_y(self, y):
        self._loss = _losses.squared_loss()
        self.n_classes = y.shape[1] if len(y.shape) == 2 else 1
        return y

    def predict(self, X):
        return self.decision_function(X)

    def predict_stage(self, X):
        preds = np.ones_like(self._models[0].predict(X))*self.intercept

        for model, step in zip(self._models, self.steps):
            preds += model.predict(X) * step
            yield preds

    def score(self, X, y):
        """Returns the average of RMSE of all outputs."""
        X = X.astype(np.float32)
        y = y.astype(np.float32)
        pred = self.predict(X)
        output_errors = np.mean((y - pred) ** 2, axis=0)

        return np.mean(np.sqrt(output_errors))
