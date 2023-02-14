""" Gradient Boosted - Convolutional Neural Network """

# Author: Seyedsaman Emami
# Author: Gonzalo Martínez-Muñoz

# Licence: GNU Lesser General Public License v2.1 (LGPL-2.1)

import numpy as np
from Base import BaseGBCNN
from Base._losses import multi_class_loss


class GBCNN(BaseGBCNN):

    def __init__(self, config):

        super().__init__(config)

    def _validate_y(self, y):
        assert len(y.shape) == 2, "input shape is not valid!"
        y = y.astype('int32')
        self._loss = multi_class_loss()
        return y

    def predict_proba(self, X):
        return multi_class_loss().raw_predictions_to_probs(self.decision_function(X))

    def _predict(self, probs):
        return np.argmax(probs, axis=1)

    def predict(self, X):
        pred_ = self.predict_proba(X)
        pred = self._predict(pred_)
        return pred

    def predict_stage(self, X):
        raw_predictions = self._pred
        for model, step in zip(self._models, self.steps):
            raw_predictions += model.predict(X) * step
            probs = multi_class_loss().raw_predictions_to_probs(raw_predictions)
            pred = np.argmax(probs, axis=1)
            yield pred

    def score(self, X, y):
        assert len(y.shape) == 2, "input shape is not valid"
        y = [np.argmax(yy, axis=None, out=None) for yy in y]
        score = y == self.predict(X)
        return np.mean(score)
