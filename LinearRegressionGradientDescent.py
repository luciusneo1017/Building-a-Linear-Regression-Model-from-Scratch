import numpy as np

class LinearRegressionGD:
    def __init__(self):       
        self.b0 = 0
        self.b1 = 0
        self.cost_history = []

    def fit(self,X, y, learning_rate, epochs, tolerance):
        n = len(X)
        self.cost_history = []

        for epoch in range(epochs):
            y_pred = self.b0 + self.b1 * X
            error = y - y_pred

            db0 = - (2/n) * np.sum(error)
            db1 = - (2/n) * np.sum(error * X)

            self.b0 -= learning_rate * db0
            self.b1 -= learning_rate * db1

            cost = (1/n) * np.sum(error ** 2)
            self.cost_history.append(cost)
            if epoch > 0 and abs(self.cost_history[-2] - cost) < tolerance:
                break

    def predict(self,X):
        y_pred = self.b0 + self.b1 * X
        return y_pred
    
    def get_params(self):
        return self.b0, self.b1