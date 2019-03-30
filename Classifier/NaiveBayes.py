import numpy as np
import pandas as pd
from sklearn.preprocessing import binarize, LabelBinarizer
from sklearn.utils.extmath import safe_sparse_dot
from scipy.sparse import issparse
from collections import defaultdict
from utils.tools import logsumexp, _check_partial_fit_first_call, label_binarize, _num_samples
import warnings

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


    def _joint_log_likelihood(self, X):
        """

        :param X:
        :return:
        """

        joint_log_likelihood = []

        for i in range(np.size(self.classes_)):
            log_prior = np.log(self.classes_)
            log_likelihood = -0.5 * np.sum(np.log(2 * np.pi * self.sigma_[i,:]))
            log_likelihood -= 0.5 * np.sum( (X - self.theta_[i, :]) ** 2 / self.sigma_[i, :], 1)

            joint_log_likelihood.append((log_prior + log_likelihood))

        return np.array(joint_log_likelihood).T


_ALPHA_MIN = 1e-10

# todo: gaussian NB model is updated by computing new mean and sigma, given new batch of labelled data;
# todo: how is it done in Multinomial NB ???


class BaseDiscreteNB(BaseNB):
    """
    For discrete / categorical data
    """

    def _update_class_log_prior(self, class_prior=None):
        n_classes = len(self.classes_)

        if class_prior is not None:
            if n_classes != len(class_prior):
                raise ValueError("Number of priors doesn't match number of classes! ")
            self.class_log_prior = np.log(class_prior)

        elif self.fit_prior:
            self.class_log_prior_ = np.log(self.class_count_) - np.log(self.class_count_.sum())

        else:
            self.class_log_prior_ = np.full(n_classes, -np.log(n_classes))

    def _check_alpha(self):
        if np.min(self.alpha) < 0:
            raise ValueError('Smoothing parameter alpha = %.1e. '
                             'alpha should be > 0.' % np.min(self.alpha))
        if isinstance(self.alpha, np.ndarray):
            if not self.alpha.shape[0] == self.feature_count_.shape[1]:
                raise ValueError("alpha should be a scalar or a numpy array "
                                 "with shape [n_features]")
        if np.min(self.alpha) < _ALPHA_MIN:
            warnings.warn('alpha too small will result in numeric errors, '
                          'setting alpha = %.1e' % _ALPHA_MIN)
            return np.maximum(self.alpha, _ALPHA_MIN)
        return self.alpha


    def partial_fit(self, X, y, classes=None, sample_weight=None):
        """
        Incremental fit on a batch of samples.

        :param X: array-like, sparse matrix, shape = [n_samples, n_features]
        :param y: array-like, shape = [n_samples]
        :param classes:  array-like, shape = [n_classes] (default=None)
        :param sample_weight:  array-like, shape = [n_samples] (default=None)
        :return:
        """

        n_samples, n_features = X.shape

        if _check_partial_fit_first_call(self, classes):
            # first call, so initialize a bunch of stuff
            n_effective_classes = len(classes) if len(classes) > 1 else 2
            self.class_count_ = np.zeros(n_effective_classes, dtype=np.float64)
            self.feature_count_ = np.zeros((n_effective_classes, n_features), dtype=np.float64)

        elif n_features != self.coef_.shape[1]:
            msg = "Number features %d does not match previous data %d"
            raise ValueError(msg%(n_features, self.coef_.shape[1]))


        Y = label_binarize(y, classes=self.class_count_)  #todo:
        if Y.shape[1] == 1:
            Y = np.concatenate((1 - Y, Y), axis=1)

        n_samples, n_classes = Y.shape

        if X.shape[0] != Y.shape[0]:
            msg = "X.shape[0]=%d and y.shape[0]=%d are incompatible."
            raise ValueError(msg % (X.shape[0], y.shape[0]))


        Y = Y.astype(np.float64)
        if sample_weight is not None:
            sample_weight = np.atleast_2d(sample_weight)
            Y *= sample_weight.T

        class_prior = self.class_prior

        self._count(X, Y) # todo: learn from each individual implementation

        alpha = self._check_alpha()
        self._update_feature_log_prob(alpha)
        self._update_class_log_prior(class_prior=class_prior)
        return self


    def fit(self, X, y, sample_weight=None):
        """
        Fit the Naive Bayes classifier ccording to X, y
        :param X: array-like, sparse matrix, shape = [n_samples, n_features]
        :param y: array-like, shape = [n_samples]
        :param sample_weight: array-like, shape = [n_samples], (default=None)
        :return: self, object
        """

        n_samples, n_features = X.shape

        labelbin = LabelBinarizer()
        Y = labelbin.fit_transform(y)
        self.classes_ = labelbin.classes_
        if Y.shape[1] == 1:
            Y = np.concatenate((1 - Y, Y), axis=1)

        Y = Y.astype(np.float64)
        if sample_weight is not None:
            sample_weight = np.atleast_2d(sample_weight)
            Y *= sample_weight.T

        class_prior = self.class_prior


        # count raw events from data
        n_effective_classes = Y.shape[1]
        self.class_count_ = np.zeros(n_effective_classes, dtype=np.float64)
        self.feature_count_ = np.zeros((n_effective_classes, n_features), dtype=np.float64)

        self._count(X,Y)
        alpha = self._check_alpha()
        self._update_feature_log_prob(alpha)
        self._update_class_log_prior(class_prior=class_prior)

        return self


    def _get_coef(self):
        return (self.feature_log_prob_[1:] if len(self.classes_) == 2 else self.feature_log_prob_)

    def _get_intercept(self):
        return (self.class_log_prior_[1:] if len(self.classes_) == 2 else self.class_log_prior_)

    coef_ = property(_get_coef)
    intercept_ = property(_get_intercept)



class MultinomialNB(BaseDiscreteNB):
    """
    Naive Bayes classifier for multinomial models

    The multinomial Naive Bayes classifier is suitable for classification with
    discrete features (e.g., word counts for text classification). The
    multinomial distribution normally requires integer feature counts. However,
    in practice, fractional counts such as tf-idf may also work.

    """

    def __init__(self, alpha=1.0, fit_prior=True, class_prior=None):
        self.alpha = alpha
        self.fit_prior = fit_prior
        self.class_prior = class_prior

    def _count(self, X, Y):
        """
        Count and smooth feature occurrences.
        :param X:
        :param Y:
        :return:
        """

        if np.any((X.data if issparse(X) else X) < 0):
            raise ValueError("Input X must be non-negative")

        self.feature_count_ += safe_sparse_dot(Y.T, X) # n_classes by n_features,
                                                       # self.feature_count_[i][j] = count of feature j in class i
        self.class_count_ += Y.sum(axis=0)


    def _update_feature_log_prob(self, alpha):
        """
        Applying smoothing to raw counts and recompute log probabilities.
        :param alpha:
        :return:
        """
        smoothed_fc = self.feature_count_ + alpha  # feature count
        smoothed_cc = smoothed_fc.sum(axis=1)      # cumulative count

        self.feature_log_prob_ = (np.log(smoothed_fc)) - np.log(smoothed_cc.reshape(-1,1))


    def _joint_log_likelihood(self, X):
        """ Computes posterior log probability of the samples X """
        return ( safe_sparse_dot(X, self.feature_log_prob_.T) + self.class_log_prior_ )



class ComplementNB(BaseDiscreteNB):
    """
    Complement Naive Bayes; designed to correct "severe assumptions" made by multinomial Naive Bayes.
    It is good gor imbalanced data sets

    parameters
    ----------
    alpha: float, additive smoothing parameter (0 for no smoothing)

    fit_prior: boolean, optional
            Only used in edge case with a single class in the training set

    ...

    """

    def __init__(self, alpha=1.0, fit_prior=True, class_prior= None, norm=False):
        self.alpha = alpha
        self.fit_prior = fit_prior
        self.class_prior = class_prior
        self.norm = norm


    def _count(self, X, Y):
        """ Count feature Occurrences """

        self.feature_count_ += safe_sparse_dot(Y.T, X) # n_classes by n_features,
                                                       # self.feature_count_[i][j] = count of feature j in class i

        self.class_count_ += Y.sum(axis=0)
        self.feature_all_ = self.feature_count_.sum(axis=0)


    def _update_feature_log_prob(self, alpha):
        """ Apply smoothing to raw counts and compute the weights """

        comp_count = self.feature_all_ + alpha - self.feature_count_
        logged = np.log(comp_count / comp_count.sum(axis=1, keepdims=True))

        # todo: not really popular model?
        pass

    def _joint_log_likelihood(self, X):
        pass



class BernoulliNB(BaseDiscreteNB):
    """
    Naive Bayes classifier for multivariate Bernoulli models.

    Like MultinomialNB, this classifier is suitable for discrete data.
    MultinomialNB works with occurrence counts vs. BernoulliNB is designed for binary/boolean features.
    """

    def __init__(self, alpha= 0.1, binarize=.0, fit_prior=True, class_prior=None):
        self.alpha = alpha
        self.binarize = binarize
        self.fit_prior = fit_prior
        self.class_prior = class_prior

    def _count(self, X, Y):
        """ Count and smooth feature occurrences """

        if self.binarize is not None:
            X = binarize(X, threshold = self.binarize)

        self.feature_count_ += safe_sparse_dot(Y.T, X)
        self.class_count_ += Y.sum(axis=0)


    def _update_feature_log_prob(self, alpha):
        """Apply smoothing to raw counts and recompute log probabilities"""

        smoothed_fc = self.feature_count_ + alpha
        smoothed_cc = self.class_count_ + alpha * 2
        self.feature_log_prob_ = ( np.log(smoothed_fc) - np.log(smoothed_cc.reshape(-1, 1)) )

    def _joint_log_likelihood(self, X):

        if self.binarize is not None:
            X = binarize(X, threshold = self.binarize)

        n_classes, n_features = self.feature_log_prob_.shape
        n_samples, n_features_X = X.shape





















