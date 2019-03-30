import numpy as np
import sklearn




class LogisticRegression(object):

    def __init__(self, learning_rate=1e-4, max_iter=1e4):

        self.learning_rate = learning_rate
        self.max_iter = max_iter


    def __sigmoid(self, x):
        return 1.0 / (1 + np.exp(-x))



    def fit(self, X, Y):

        n_samples, n_features = X.shape

        pass



    def predict(self, X):
        pass