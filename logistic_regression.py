######################################################
# Diabetes Prediction with Logistic Regression
######################################################

# Problem Statement:
# Can we develop a machine learning model that predicts whether a person has diabetes based on given features?

# The dataset is part of the National Institute of Diabetes and Digestive and Kidney Diseases in the U.S.
# It consists of medical data collected from Pima Indian women in Phoenix, Arizona, aged 21 and above.
# It contains 768 observations and 8 numerical independent variables.
# The target variable "Outcome" indicates whether the diabetes test result is positive (1) or negative (0).

# Features:
# Pregnancies: Number of pregnancies
# Glucose: Glucose level
# BloodPressure: Blood pressure
# SkinThickness: Skin thickness
# Insulin: Insulin level
# BMI: Body Mass Index
# DiabetesPedigreeFunction: Function estimating diabetes risk based on family history
# Age: Age (years)
# Outcome: Indicates whether the person has diabetes (1) or not (0)


# PROJECT STEPS:
# 1. Exploratory Data Analysis
# 2. Data Preprocessing
# 3. Model & Prediction
# 4. Model Evaluation
# 5. Model Validation: Holdout
# 6. Model Validation: 10-Fold Cross Validation
# 7. Prediction for a New Observation

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report, plot_roc_curve
from sklearn.model_selection import train_test_split, cross_validate

pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

# Function to calculate outlier thresholds
def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

# Function to check if a variable has outliers
def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    return dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None)

# Function to replace outliers with threshold values
def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


#############################################
# 1. Exploratory Data Analysis
#############################################

df = pd.read_csv("datasets/diabetes.csv")
df.head()

df.shape

########################
# Target Variable Analysis
########################

df["Outcome"].value_counts()

sns.countplot(x="Outcome", data=df)
plt.show()

# Percentage distribution of classes
100 * df["Outcome"].value_counts() / len(df)

########################
# Feature Analysis
########################

df.describe().T

# Visualizing numerical variables
def plot_numerical_col(dataframe, numerical_col):
    dataframe[numerical_col].hist(bins=20)
    plt.xlabel(numerical_col)
    plt.show()

cols = [col for col in df.columns if "Outcome" not in col]

for col in cols:
    plot_numerical_col(df, col)

###########################
# Target vs Features
###########################

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n")

for col in cols:
    target_summary_with_num(df, "Outcome", col)


################################################
# 2. Data Preprocessing
################################################

df.isnull().sum()

df.describe().T

# Checking for outliers
for col in cols:
    print(col, check_outlier(df, col))

# Replacing outliers in the Insulin variable
replace_with_thresholds(df, "Insulin")

###################
# Standardization
###################

for col in cols:
    df[col] = RobustScaler().fit_transform(df[[col]])

df.head()


#####################################################
# 3. Model & Prediction
#####################################################

y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)

# Training the logistic regression model
log_model = LogisticRegression().fit(X, y)

# Model parameters
log_model.intercept_
log_model.coef_

# Predictions
y_pred = log_model.predict(X)

y[0:10]
y_pred[0:10]


####################################################
# 4. Model Evaluation
####################################################

def plot_confusion_matrix(y, y_pred):
    acc = round(accuracy_score(y, y_pred), 2)
    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot=True, fmt=".0f")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Accuracy Score: {0}".format(acc), size=10)
    plt.show()

plot_confusion_matrix(y, y_pred)

print(classification_report(y, y_pred))

# ROC AUC Score
y_prob = log_model.predict_proba(X)[:, 1]
roc_auc_score(y, y_prob)


#####################################################
# 5. Model Validation: Holdout
#####################################################

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=17)

# Training the model on train set
log_model = LogisticRegression().fit(X_train, y_train)

# Predicting on test set
y_pred = log_model.predict(X_test)

# Class probabilities
y_prob = log_model.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred))

# ROC Curve
plot_roc_curve(log_model, X_test, y_test)
plt.title("ROC Curve")
plt.plot([0, 1], [0, 1], "r--")
plt.show()

# AUC Score
roc_auc_score(y_test, y_prob)


#######################################################
# 6. Model Validation: 10-Fold Cross Validation
#######################################################

y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)

log_model = LogisticRegression().fit(X, y)

cv_results = cross_validate(log_model, X, y, cv=5, scoring=["accuracy", "precision", "recall", "f1", "roc_auc"])

cv_results["test_accuracy"].mean()
cv_results["test_precision"].mean()
cv_results["test_recall"].mean()
cv_results["test_f1"].mean()
cv_results["test_roc_auc"].mean()


#############################################
# 7. Prediction for A New Observation
#############################################

X.columns

random_user = X.sample(1, random_state=45)

# Prediction
log_model.predict(random_user)
