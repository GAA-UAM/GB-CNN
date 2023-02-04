""" Loss functions for Gradient Boosted Convolutional Neural Network """

# Author: Seyedsaman Emami
# Author: Gonzalo Martínez-Muñoz

# Licence: GNU Lesser General Public License v2.1 (LGPL-2.1)

import numpy as np

from abc import abstractmethod
from scipy.special import logsumexp


class loss:
    """ Template class for the loss function """

    @abstractmethod
    def model0(self, y):
        """abstract method for initialization of approximation"""

    @abstractmethod
    def derive(self, y, prev):
        """abstract method for derive"""

    @abstractmethod
    def newton_step(self, y, residuals, new_predictions):
        """abstract method for newton raphson step"""

    @abstractmethod
    def __call__(self, y, pred):
        """call"""


class classification_loss(loss):
    """ Base class for classification losses """

    @abstractmethod
    def raw_predictions_to_probs(self, preds):
        """abstract method for raw_predictions_to_probs"""

class multi_class_loss(classification_loss):
    """ Entropy loss por multi-class classification tasks """

    def model0(self, y):
        return np.zeros_like(y[0, :])

    def derive(self, y, prev):
        return y - np.nan_to_num((np.exp(prev -
                                         logsumexp(prev, axis=1, keepdims=True))))

    def newton_step(self, y, residuals, new_predictions):
        f_m = new_predictions
        p = y-residuals
        return -np.sum(f_m * (y - p)) / np.sum(f_m * f_m * p * (p - 1))

    def raw_predictions_to_probs(self, preds):
        return np.exp(preds - logsumexp(preds, axis=1, keepdims=True))

    def __call__(self, y, pred):
        return np.sum(-1 * (y * pred).sum(axis=1) +
                      logsumexp(pred, axis=1))
