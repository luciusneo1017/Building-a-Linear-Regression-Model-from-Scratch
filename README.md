# Building a Linear Regression Model from Scratch (Gradient Descent + Closed Form)

This project walks through building **simple linear regression** in Python without relying on scikit-learn’s training routines.  
I implement two approaches:

- **Gradient Descent (iterative optimization)**  
- **Closed-Form / Normal Equation (analytical solution)**  

Finally, the learned parameters are **cross-checked against scikit-learn** for verification.

## Models

- `LinearRegressionGD`: linear regression trained via gradient descent  
  - Tracks `cost_history` (MSE-style loss)
  - Includes early stopping using a `tolerance` threshold
- `LinearRegressionCF`: linear regression trained via the normal equation  
- Plots:
  - Salary vs Years of Experience scatter plot
  - Fitted regression line overlay
- Validation:
  - Compares intercept and slope vs `sklearn.linear_model.LinearRegression`


