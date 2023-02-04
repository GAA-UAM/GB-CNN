""" Gradient Boosted - Convolutional Neural Network """

# Author: Seyedsaman Emami
# Author: Gonzalo Martínez-Muñoz

# Licence: GNU Lesser General Public License v2.1 (LGPL-2.1)

import numpy as np
from Base import BaseEstimator
from Base._losses import multi_class_loss


class GBCNN(BaseEstimator):

    def __init__(self, config):

        super().__init__(config)

    def predict_proba(self, X):
        return multi_class_loss().raw_predictions_to_probs(self.decision_function(X))

    def _predict(self, probs):
        return np.argmax(probs, axis=1)

    def predict(self, X):
        pred_ = self.predict_proba(X)
        pred = self._predict(pred_)
        print (np.unique(pred))
        return pred

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
