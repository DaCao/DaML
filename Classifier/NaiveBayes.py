import numpy as np
import pandas as pd
from collections import defaultdict
from utils.tools import logsumexp, _check_partial_fit_first_call

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

    @staticmethod
    def _update_mean_variance(n_past, mu, var, X, sample_weight=None):
        """
        Given starting sample count, mean, and variance, a new set of
        points X, and optionally sample weights, return the updated mean and
        variance. (NB - each dimension (column) in X is treated as independent
        -- you get variance, not covariance).

        :param n_past: int; Number of samples represented in old mean and variance.
        :param mu: array-like, shape (number of Gaussians,);  Means for Gaussians in original set.
        :param var: array-like, shape (number of Gaussians,);  Variances for Gaussians in original set.
        :param X:
        :param sample_weight:
        :return total_mu:  updated mean for each gaussian with new data
        :return total_var: updated var for each gaussian with new data
        """

        if X.shape[0] == 0:
            return mu, var

        # compute mean and variance of new data
        if sample_weight is None:
            n_new = X.shape[0]
            new_mu = np.mean(X, axis=0)
            new_var = np.var(X, axis=0)
        else:
            pass # todo: later

        if n_past == 0:
            return new_mu, new_var

        n_total = n_past + n_new

        # updated total mean
        total_mu = (n_past * mu + n_new * new_mu) / n_total

        # Combine variance of old and new data.  This is achieved by combining the sum-of-squared-differences (ssd)
        # references:
        # https://www.geeksforgeeks.org/find-combined-mean-variance-two-series/
        # https://www.emathzone.com/tutorials/basic-statistics/combined-variance.html
        old_ssd = var * n_past
        new_ssd = new_var * n_new
        total_ssd = (old_ssd + new_ssd + (n_past / float(n_new * n_total)) * (n_new * mu - n_new * new_mu) ** 2)
        total_var = total_ssd / n_total

        return total_mu, total_var



    def partial_fit(self, X, y, classes=None, sample_weight=None):
        """
        Incremental fit on a batch of samples.
        This method is expected to be called several times consecutively on different chunks of a dataset so as to
        implement out-of-core or online learning. This is especially useful when the whole dataset is too big to fit
        in memory at once.

        :param X: array-like, shape (n_samples, n_features)
        :param y: array-like, shape (n_samples,)
        :param classes: array-like, shape (n_classes,), optional (default=None)
                        List of all the classes that can possibly appear in the y vector.
        :param sample_weight:
        :return:
        """
        return self._partial_fit(X, y, classes, _refit=False, sample_weight=sample_weight)


    def _partial_fit(self, X, y, classes=None, _refit=False, sample_weight=None):
        """

        Implementation of Gaussian Naive Bayes fitting

        :param X: array-like, shape (n_samples, n_features)
        :param y: array-like, shape (n_samples,)
        :param classes: array-like, shape (n_classes,), optional (default=None)
                        List of all the classes that can possibly appear in the y vector.
        :param _refit:  if True, act as if this is the first time called
        :param sample_weight: array-like, shape (n_samples,), optional (default=None)
                              Weights applied to individual samples (1. for unweighted).
        :return:
        """

        if _refit:
            self.classes_ = None

        # boost ratio of data variance between dimensions(features) by epsilon, a small fraction of the standard
        # deviation of the largest variance across features.
        self.epsilon_ = self.smoothing * np.var(X, axis=0).max()

        if _check_partial_fit_first_call(self, classes):
            # first call to _partial_fit, so we do some initializations:
            # counters
            num_features, num_classes = X.shape[0], len(self.classes_)
            self.theta_, self.sigma_ = np.zeros((num_classes,num_features))
            self.class_count_ = np.zeros(num_classes, dtype=np.float64)

            # class priors
            if self.priors is not None:
                priors = np.asarray(self.priors)

                # provided prior should match number of classes
                if len(priors) != num_classes:
                    raise ValueError("Number of priors must match number of classes. ")

                # prior must sum to 1.0; and non negative
                if not np.isclose(sum(priors), 1.0) or (priors < 0).any():
                    raise ValueError("Invalid prior! prior must be non-negative and sum to 1.0 ")

                self.class_prior_ = priors

            else:
                # Initialize priors to 0 for every class
                self.class_prior_ = np.zeros(num_classes, dtype=np.float64)

        else:
            if X.shape[1] != self.theta_.shape[1]:
                raise ValueError("Number of features %d does not match previous data %d "%(X.shape[1], self.theta_.shape[1]))

            self.sigma_[:, :] -= self.epsilon_



        classes = self.classes_
        unique_y = np.unique(y)
        unique_y_in_classes = np.in1d(unique_y, classes)

        if not np.all(unique_y_in_classes):
            raise ValueError("label(s) %s in y does not exist in initial classes %s"%(unique_y[~unique_y_in_classes], classes))


        for y_i in unique_y: # for a class
            i = classes.searchsorted(y_i)
            X_i = X[y == y_i, :]

            if sample_weight is None:
                sw_i = None
                N_i = X_i.shape[0]
            else:
                # sw_i =
                # N_i =
                pass  # todo:


            new_theta, new_sigma = self._update_mean_variance(self.class_count_[i], self.theta_[i,:],
                                                              self.sigma_[i,:], X_i, sw_i)

            self.theta_[i, :] = new_theta
            self.sigma_[i, :] = new_sigma
            self.class_count_[i] += N_i

        self.sigma_[:, :] += self.epsilon_

        if self.priors is None:
            self.priors = self.class_count_ / self.class_count_.sum()

        return self













