import numpy as np


class Perceptron(object):
    def __init__(self, rate=0.01, iter_count=50, state=1):
        self.rate = rate
        self.iter_count = iter_count
        self.state = state
        print("Initialized Perceptron")

    def net_input(self, X):
        # Add bias weight
        return np.dot(X, self.weights[1:]) + self.weights[0]
    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)
    def fit(self, X, y):
        rgen = np.random.RandomState(self.state)
        self.weights = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])

        self.errors = []

        for _ in range(self.iter_count):
            errors = 0
            for xi, output in zip(X, y):
                update = self.rate * (output - self.predict(xi))
                self.weights[1:] += update * xi
                self.weights[0] += update
                errors += int(update != 0.0)
            self.errors.append(errors)
        return self

perceptron = Perceptron()
