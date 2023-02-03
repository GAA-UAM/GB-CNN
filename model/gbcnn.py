""" Deep - Gradient Boosted Neural Network """

# Author: Seyedsaman Emami
# Author: Gonzalo Martínez-Muñoz

# Licence: GNU Lesser General Public License v2.1 (LGPL-2.1)

import numpy as np
from Base._losses import multi_class_loss
from Base import BaseMultilayer
from sklearn.utils.multiclass import check_classification_targets



class GBCNN(BaseMultilayer):

    def __init__(self,
                 iter=10,
                 eta=0.1,
                 learning_rate=1e-3,
                 total_nn=200,
                 num_nn_step=1,
                 batch_size=128,
                 early_stopping=10,
                 verbose=False,
                 random_state=None):

        super().__init__(iter,
                         eta,
                         learning_rate,
                         total_nn,
                         num_nn_step,
                         batch_size,
                         early_stopping,
                         verbose,
                         random_state)

    def _validate_y(self, y):

        check_classification_targets(y)
        y = y.astype('int32')
        return y

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

