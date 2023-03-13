""" Gradient Boosted - Deep Neural Network """

# Author: Seyedsaman Emami
# Author: Gonzalo Martínez-Muñoz

# Licence: GNU Lesser General Public License v2.1 (LGPL-2.1)

import numpy as np
from Base import _losses
from Base._base import BaseGBDNN
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils.multiclass import type_of_target, check_classification_targets


class GBDNNClassifier(BaseGBDNN):

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
