# First perceptron implementation...

import numpy as np

class Perceptron(object):
    """ Perceptron Classifier

    parameters
    ----------------------
    learning_rate : float
        Learning Rate (between 0 and 1)
    n_length_iterator : int
        Number of passes over dataset
    random_state : int
        random num generator for init of weights

    attributes
    ---------------------
    w_ 1d-array
        Weights after fitting
    errors : list
        Number of misclassifications (updates) in each epoch

    """

    def __init__(self, learning_rate=0.01, n_length_iterator=50, random_state=1):
        self.learning_rate = learning_rate
        self.n_length_iterator = n_length_iterator
        self.random_state = random_state

    def net_input(self, row):
        """Calculate net input"""
        return np.dot(row, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.net_input(X) >- 0.0, 1, -1)
    
    def fit(self, X, y):
        """ Fit training data.

        params
        ---------------
        X : {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples]
            Target values.
        
        returns
        ------
        self : object
        
        """

        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1]) 

        self.errors_ = []

        for _ in range(self.n_length_iterator):
            errors = 0
            for xi, target in zip(X, y):
                update = self.learning_rate * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors) # Track the incorrect decisions for each iteration over the dataset...
        return self
