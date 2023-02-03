""" Gradient Boosted - Convolutional Neural Network """

# Author: Seyedsaman Emami
# Author: Gonzalo Martínez-Muñoz

# Licence: GNU Lesser General Public License v2.1 (LGPL-2.1)

import numpy as np
from Base import BaseMultilayer
from Base._losses import multi_class_loss


class GBCNN(BaseMultilayer):

    def __init__(self, config):

        super().__init__(config)

    def predict_proba(self, X):
        return multi_class_loss().raw_predictions_to_probs(self.decision_function(X))

    def _predict(self, probs):
        return np.argmax(probs, axis=1)

    def predict(self, X):
        pred = self.predict_proba(X)
        return self._predict(pred)

    def predict_stage(self, X):
        raw_predictions = self._pred
        for model, step in zip(self._models, self.steps):
            raw_predictions += model.predict(X) * step
            probs = multi_class_loss().raw_predictions_to_probs(raw_predictions)
            pred = np.argmax(probs, axis=1)
            yield pred

    def score(self, X, y):
        score = y == self.predict(X)
        return np.mean(score)
