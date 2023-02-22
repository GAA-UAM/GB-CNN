""" Loss functions for Deep Gradient Boosted Neural Networks """

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


class squared_loss(loss):
    """ Squared loss for regression problems """

    def model0(self, y):
        return np.ones(1)*np.mean(y)

    def derive(self, y, prev):
        return y-prev

    def newton_step(self, y, residuals, new_predictions):
        return 1

    def __call__(self, y, pred):
        return (y-pred)**2


class classification_loss(loss):
    """ Base class for classification losses """

    @abstractmethod
    def raw_predictions_to_probs(self, preds):
        """abstract method for raw_predictions_to_probs"""


class log_exponential_loss(classification_loss):
    """ Log-exponential loss for binary classification tasks """

    def model0(self, y):
        ymed = np.mean(y)
        return np.ones(1)*(0.5 * np.log((1+ymed) / (1-ymed)))

    def derive(self, y, prev):
        return np.nan_to_num(2.0 * y / (1 + np.exp(2.0 * y * prev)))

    def newton_step(self, y, residuals, new_predictions):
        f_m = np.squeeze(new_predictions)
        return np.sum(residuals * f_m) / np.sum(residuals * f_m * f_m * (2.0 * y - residuals))

    def raw_predictions_to_probs(self, preds):
        preds = np.squeeze(1 / (1 + np.exp(-2 * preds)))
        return np.vstack((1-preds, preds)).T

    def __call__(self, y, pred):
        return np.log(1 + np.exp(-2.0 * y * pred))


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
        return (-1 * (y * pred).sum(axis=1) + logsumexp(pred, axis=1))
