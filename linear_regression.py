##################################################
# Sales Prediction with Linear Regression
##################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option("display.float_format", lambda x: "%.2f" % x)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score

######################################################
# Simple Linear Regression with OLS Using Scikit-Learn
######################################################

df = pd.read_csv("datasets/advertising.csv")
df.head()
df.shape

X = df[["TV"]]
y = df[["sales"]]

#########################
# MODEL
#########################

reg_model = LinearRegression().fit(X, y)

"""
Single variable regression:
    y_hat = b + w*x
For this case:  
    y_hat = b + w*TV
"""

# Intercept (b)
reg_model.intercept_[0]

# Coefficient of TV (w1)
reg_model.coef_[0][0]

#########################
# PREDICTION
#########################

# Prediction for 150 units of TV spending
reg_model.intercept_[0] + reg_model.coef_[0][0] * 150  

# Prediction for 500 units of TV spending
reg_model.intercept_[0] + reg_model.coef_[0][0] * 500  

df.describe().T

# Visualizing the Model
g = sns.regplot(x=X,  
                y=y,  
                scatter_kws={"color": "b", "s": 9},  
                ci=False,  
                color="r")  
g.set_title(f"Model Equation: Sales = {round(reg_model.intercept_[0], 2)} + TV*{round(reg_model.coef_[0][0], 2)}")
g.set_ylabel("Sales")
g.set_xlabel("TV Spending")
plt.xlim(-10, 310)
plt.ylim(bottom=0)
plt.show()

#######################
# Model Performance
#######################

# Predicted values
y_pred = reg_model.predict(X)

# MSE
mean_squared_error(y, y_pred)  

# RMSE
np.sqrt(mean_squared_error(y, y_pred))

# MAE
mean_absolute_error(y, y_pred)

# R-Squared Score
reg_model.score(X, y)  

"""
R-Squared:
    Measures how much of the variance in the dependent variable can be explained by independent variables.
"""

###########################################
# Multiple Linear Regression
###########################################

df = pd.read_csv("datasets/advertising.csv")

# Independent variables
X = df.drop("sales", axis=1)

# Dependent variable
y = df[["sales"]]

###################
# Model
###################

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.20,  
                                                    random_state=1)

# Train the model
reg_model = LinearRegression()
reg_model.fit(X_train, y_train)

# Intercept (b)
reg_model.intercept_

# Coefficients (w)
reg_model.coef_

#################
# Prediction
#################

# Predicting sales for new data
new_data = [[30], [10], [40]]
new_data = pd.DataFrame(new_data).T
reg_model.predict(new_data)

########################
# Model Evaluation
########################

# Train RMSE
y_pred = reg_model.predict(X_train)
np.sqrt(mean_squared_error(y_train, y_pred))  

# Train R-Squared Score
reg_model.score(X_train, y_train)  

# Test RMSE
y_pred = reg_model.predict(X_test)  
np.sqrt(mean_squared_error(y_test, y_pred))  

# Test R-Squared Score
reg_model.score(X_test, y_test)  

# 10-Fold Cross Validation RMSE
np.mean(np.sqrt(-cross_val_score(reg_model,  
                                 X,  
                                 y,  
                                 cv=10,  
                                 scoring="neg_mean_squared_error")))

# 5-Fold Cross Validation RMSE
np.mean(np.sqrt(-cross_val_score(reg_model,  
                                 X,  
                                 y,  
                                 cv=5,  
                                 scoring="neg_mean_squared_error")))

###########################################################
# SIMPLE LINEAR REGRESSION with GRADIENT DESCENT from scratch
###########################################################

# Cost Function (MSE)
def cost_function(Y, b, w, X):
    """
    Calculate MSE

    Parameters
    ----------
    Y : dependent variable
    b : bias
    w : weight
    X : independent variable

    Returns
    -------
    mse : mean squared error
    """
    m = len(Y)  
    sse = 0

    for i in range(m):  
        y_hat = b + w * X.iloc[i]  
        y = Y.iloc[i]  
        sse += (y_hat - y) ** 2  

    mse = sse / m  
    return mse

# Update Weights
def update_weights(Y, b, w, X, learning_rate):
    """
    Gradient descent weight update

    Parameters
    ----------
    Y : dependent variable
    b : bias
    w : weights
    X : independent variable
    learning_rate : step size in gradient descent

    Returns
    -------
    new_b :
    new_w :
    """
    m = len(Y)  

    b_deriv_sum = 0
    w_deriv_sum = 0

    for i in range(m):  
        y_hat = b + w * X.iloc[i]  
        y = Y.iloc[i]  
        b_deriv_sum += (y_hat - y)  
        w_deriv_sum += (y_hat - y) * X.iloc[i]  

    new_b = b - (learning_rate * (1 / m) * b_deriv_sum)  
    new_w = w - (learning_rate * (1 / m) * w_deriv_sum)  

    return new_b, new_w


# Train function
def train(Y, initial_b, initial_w, X, learning_rate, num_iters):
    """
    Train the model using gradient descent

    Parameters
    ----------
    Y : dependent variable
    initial_b : initial bias value
    initial_w : initial weight value
    X : independent variable(s)
    learning_rate
    num_iters : number of iterations

    Returns
    -------
    cost_history
    b
    w
    """
    print("Starting gradient descent at b = {0}, w = {1}, mse = {2}".format(initial_b, initial_w,
                                                                           cost_function(Y, initial_b, initial_w, X)))

    b = initial_b
    w = initial_w
    cost_history = []

    for i in range(num_iters):  
        b, w = update_weights(Y, b, w, X, learning_rate)  
        mse = cost_function(Y, b, w, X)  
        cost_history.append(mse)

        if i % 100 == 0:  
            print("iter={:d}    b={:.2f}    w={:.4f}    mse={:.4}".format(i, b, w, mse))

    print("After {0} iterations b = {1}, w = {2}, mse = {3}".format(num_iters, b, w, cost_function(Y, b, w, X)))
    return cost_history, b, w


df = pd.read_csv("datasets/advertising.csv")

X = df["radio"]
Y = df["sales"]

# Hyperparameters
learning_rate = 0.001
initial_b = 0.001
initial_w = 0.001
num_iters = 1000

cost_history, b, w = train(Y, initial_b, initial_w, X, learning_rate, num_iters)
