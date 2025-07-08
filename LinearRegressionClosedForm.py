import numpy as np

class LinearRegressionCF:
    def __init__(self):
        self.coefficients = None
    def fit(self,X,y):
        ones = np.ones((X.shape[0],1))
        X_b = np.hstack((ones,X.to_numpy().reshape(-1,1)))

        # Normal Equation: β = (XᵀX)⁻¹ Xᵀy
        XTX = X_b.T.dot(X_b)
        XTy = X_b.T.dot(y)
        self.coefficients = np.linalg.inv(XTX).dot(XTy)

    def predict(self):
        ones = np.ones((X.shape[0],1))
        X_b = np.hstack((ones,X.to_numpy().reshape(-1,1)))
        y_pred = X_b.dot(self.coefficients)
        return y_pred
    def get_params(self):
        b0 = self.coefficients[0]
        b1 = self.coefficients[1]
        return b0,b1