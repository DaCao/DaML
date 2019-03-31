import numpy as np
import sklearn


# ref:
# https://medium.com/@martinpella/logistic-regression-from-scratch-in-python-124c5636b8ac
# https://github.com/Benlau93/Machine-Learning-by-Andrew-Ng-in-Python/blob/master/LogisticRegression/ML_RegularizedLogisticRegression.ipynb
# http://openclassroom.stanford.edu/MainFolder/DocumentPage.php?course=MachineLearning&doc=exercises/ex5/ex5.html


# todo: use normal equations to compute??
# todo: include L1 regularization

class LogisticRegression(object):

    def __init__(self, learning_rate=1e-4, max_iter=1e4, tol=1e-4, penalty='l2',
                 lambda_=1.0, verbose=True):
        """
        
        :param learning_rate:
        :param max_iter:
        :param verbose:
        :param tol:  float; Tolerance for stopping criteria.

        :param penalty: Used to specify the norm used in the penalization.

        :param lambda_: regularization strength; must be a positive float.
                        Larger values specify stronger regularization.

        :param fit_intercept: Specifies if a constant (a.k.a. bias or intercept) should be added to the decision function.
        """

        self.alpha = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.penalty = penalty
        self.lambda_ = lambda_
        self.verbose = verbose


    def __add_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)


    def __sigmoid(self, x):
        return 1.0 / (1 + np.exp(-x))


    def __loss(self, y, h, theta_):
        """
        compute loss (cost) with L2 regularization
        :param y: 2d array; n_samples by 1
        :param h: 2d array; n_samples by 1
        :param theta_: 2d array n_features by 1
        :return:
        """

        n_samples = y.shape[0]
        loss = (- y * np.log(h) - (1 - y) * np.log(1 - h)).mean(axis=0)

        if self.penalty == 'l2':
            # L2 regularization: 1/2 经常会看到，主要是为了后面求导的结果方便，后面那一项求导会产生一个2，与1/2相乘刚好凑整。
            reg = self.lambda_ / 2.0 * n_samples * sum(theta_**2)
        if self.penalty == 'l1':
            # todo:
            pass

        return loss + reg


    def gradient_ascent(self, X, Y):
        """
        Do gradient_descent for loss function with L2 regularization
        gradient_descent is basically the same, see:
        https://blog.csdn.net/ligang_csdn/article/details/53838743

        for more info of why gradient is calculated this way (j_0 and j_1), see:
        https://machinelearningmedium.com/2017/09/15/regularized-logistic-regression/

        :param X: 2d array;  n_samples by n_features
        :param Y: 1d array;  n_samples by 1
        :return:
        """
        n_samples, n_features = X.shape
        theta_ = np.ones((n_features, 1)) # parameters of the regression model
        h = self.__sigmoid(np.dot(X, theta_))

        prev_loss = np.inf

        for k in range(self.max_iter):
            tmp = np.dot(X.T, h - Y) / float(n_samples)
            j_0 = tmp[0]
            j_1 = tmp[1:] + (self.lambda_ / n_samples) * theta_[1:]
            grad = np.vstack((j_0[:, np.newaxis], j_1))
            theta_ -= self.alpha * grad

            h = self.__sigmoid(np.dot(X, theta_))
            loss_val = self.__loss(Y, h, theta_)

            if self.verbose and k%1000 == 0:
                print('loss at {k}th iteration is {val}'.format(k=k, val=loss_val))

            if abs(prev_loss - loss_val) / abs(prev_loss) < self.tol:
                print('loss at {k}th iteration is {val}; quit gradient descent...'.format(k=k, val=loss_val))
                break

            prev_loss = loss_val

        return theta_


    def fit(self, X, Y):
        X = self.__add_intercept(X)
        self.theta_ = self.gradient_ascent(X, Y)


    def predict_prob(self, X):
        X = self.__add_intercept(X)
        return self.__sigmoid(np.dot(X, self.theta_))


    def predict(self, X, threshold):
        return self.predict_prob(X) >= threshold


