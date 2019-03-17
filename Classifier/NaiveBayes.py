import numpy as np
import pandas as pd
from collections import defaultdict
from utils.tools import logsumexp

from sklearn



# https://github.com/Arctanxy/learning_notes/blob/master/study/machine_learning/Bayes/NavieBayes.py


class BaseNB(object):

    def __init__(self, X, Y, smoothing=0.001):

        if len(X) != len(Y):
            print('input dimensions mismatch!') #todo: error handling??
            return False

        self.smoothing = smoothing

        if not isinstance(X, np.ndarray):
            X = np.asarray(X)
        if not isinstance(Y, np.ndarray):
            Y = np.asarray(Y)
        self.X = X
        self.Y = Y
        self.n_samples, self.n_features = X.shape


    def _joint_log_likelihood(self, X):
        """
        reference: https://en.wikipedia.org/wiki/Naive_Bayes_classifier
        posterior = (prior * likelihood) / evidence
        P(c|x) = (P(x|c) * P(c)) / P(x)

        Computes posterior log probability of X; it is unnormalized by P(x), a.k.a Evidence

         I.e. for each class c, compute ``log P(c) + log P(x|c)`` for all rows x of X, as an array-like of
        shape [n_classes, n_samples].

        :param X: array-like, shape = [n_samples, n_features]
        :return: array-like, shape = [n_samples, n_classes]
        """
        pass


    def predict(self, X):
        """
        Make classification on an array of test vectors
        :param X:  array-like, shape = [n_samples, n_features]
        :return: Y_pred: array, shape = [n_samples]; Predicted target values for X
        """
        jll = self._joint_log_likelihood(X)
        return self.classes_[np.argmax(jll, axis=1)]


    def predict_log_prob(self, X):
        """
        Compute the normalized posterior log probability for test vector

        reference: https://en.wikipedia.org/wiki/Naive_Bayes_classifier
            posterior = (prior * likelihood) / evidence
            P(c|x) = (P(x|c) * P(c)) / P(x)

        :param X:  array-like, shape = [n_samples, n_features]
        :return: array-like, shape = [n_samples, n_classes]
        """

        jll = self._joint_log_likelihood(X)  # log( (P(x|c) * P(c)) )
        log_prob_x = logsumexp(jll, axis=1)  # log(x)

        # now normalize jll by log_prob_x:  log((P(x|c) * P(c))/P(x)) = log((P(x|c) * P(c))) - log(x)
        normalized_jll = jll - np.atleast_2d(log_prob_x).T   # log( P(c|x) )

        return normalized_jll


    def predict_prob(self, X):
        """
        Compute the posterior probability for test vector

        :param X:  array-like, shape = [n_samples, n_features]
        :return: array-like, shape = [n_samples, n_classes]
        """
        return np.exp(self.predict_log_prob(X))



class GaussianNB(BaseNB):
    """
    Gaussian Naive Bayes
    Can perform online updates to model parameters via `partial_fit` method.
    """

    def __init__(self, priors=None, smoothing = 1e-9):
        self.priors = priors
        self.smoothing = smoothing


    def

