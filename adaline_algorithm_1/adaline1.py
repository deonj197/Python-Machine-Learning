import numpy as np


class AdalineGD(object):
    """ADAaptive LInear NEuron classifier

    Params
    ------------
    learning_rate : float
        Learning rate (between 0.0 and 1.0)
    n_iter : int
        Passes over the training set
    random_state : int
        Random number generator seed for random weight 
        initialization.

    Attributes
    -------------
    weights_ : 1d-array
        Weights after fitting
    cost_ : list
        Sum-of-squares cost function value in each epoch.
    """

    def __init__(self, learning_rate=0.01, n_iter=50, random_state=1):
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        """ Fit training data.
        Params
        -------
        X : {array-like}, shape= [n_samples, n_features]
            Training vectors, where n_samples is the num samples
            and n_features is the num features.
        y: array-like, shape = [n_samples]
            Target values.

        Returns
        ---------
        self : object
        """
        rgen = np.random.RandomState(self.random_state)
        self.weights_ = rgen.normal(loc=0.0, scale=0.01,
                                    size=1 + X.shape[1])

        self.cost_ = []

        # TODO perform fitting of data here...

    def net_input(self, X):
        """Calc net input"""
        return np.dot(X, self.weights_[1:]) + self.weights_[0]

    def activation(self, Z):
        """Compute linear activation: identity funtion for net input"""
        return Z

    def predict(self, X):
        """Return class label after unit step"""

        # consider np.where()the threshold function...
        return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)
