############################################
# Decision Tree Classification: CART
############################################

"""
1. Exploratory Data Analysis
2. Data Preprocessing & Feature Engineering
3. Modelling using CART
4. Hyperparameter Optimization with GridSearchCV
5. Final Model
6. Feature Importance
7. Analyzing Model Complexity with Learning Curves (BONUS)
8. Visualizing the Decision Tree
9. Extracting Python/SQL/Excel Codes of Decision Rules
10. Prediction using Python Codes
11. Saving and Loading Model
"""


# Imports and display settings
import warnings
import joblib
import pydotplus
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_graphviz, export_text
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, validation_curve
from skompiler import skompile

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
warnings.simplefilter(action="ignore", category=Warning)

#################################
# 1. Exploratory Data Analysis
#################################

#################################
# 2. Data Preprocessing & Feature Engineering
#################################

#################################
# 3. Modelling using CART
#################################

df = pd.read_csv("datasets/diabetes.csv")

y = df["Outcome"]  # Dependent variable
X = df.drop(["Outcome"], axis=1)  # Independent variables

# Model creation
cart_model = DecisionTreeClassifier(random_state=1).fit(X, y)

# Predictions

y_pred = cart_model.predict(X)  # Confusion matrix prediction
y_prob = cart_model.predict_proba(X)[:, 1]  # AUC prediction

# Confusion matrix report
print(classification_report(y, y_pred))

# AUC Score
roc_auc_score(y, y_prob)

################
# 4. Hyperparameter Optimization with GridSearchCV
################

cart_params = {"max_depth": range(1, 11), "min_samples_split": range(2, 20)}

cart_best_grid = GridSearchCV(
    cart_model, cart_params, cv=5, n_jobs=-1, verbose=1
).fit(X, y)

print(cart_best_grid.best_params_)  # Best hyperparameters
print(cart_best_grid.best_score_)  # Best score

##################################
# 5. Final Model
###################################

cart_final = DecisionTreeClassifier(
    **cart_best_grid.best_params_, random_state=17
).fit(X, y)

cv_results = cross_validate(
    cart_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"]
)

print(cv_results["test_accuracy"].mean())
print(cv_results["test_f1"].mean())
print(cv_results["test_roc_auc"].mean())

###################################
# 6. Feature Importance
###################################

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({"Value": model.feature_importances_, "Feature": features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False)[0:num])
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig("feature_importance.png")

plot_importance(cart_final, X, num=5)

#######################################
# 7. Analyzing Model Complexity with Learning Curves
#######################################

def val_curve_params(model, X, y, param_name, param_range, scoring="roc_auc", cv=10):
    train_score, test_score = validation_curve(
        model, X=X, y=y, param_name=param_name, param_range=param_range, scoring=scoring, cv=cv
    )

    plt.plot(param_range, np.mean(train_score, axis=1), label="Training Score", color='b')
    plt.plot(param_range, np.mean(test_score, axis=1), label="Validation Score", color='g')
    plt.title(f"Validation Curve for {type(model).__name__}")
    plt.xlabel(f"{param_name}")
    plt.ylabel(f"{scoring}")
    plt.legend(loc='best')
    plt.show()

val_curve_params(cart_final, X, y, "max_depth", range(1, 11))
val_curve_params(cart_final, X, y, "min_samples_split", range(2, 20))

#######################################
# 8. Visualizing the Decision Tree
#######################################

def tree_graph(model, col_names, file_name):
    tree_str = export_graphviz(model, feature_names=col_names, filled=True, out_file=None)
    graph = pydotplus.graph_from_dot_data(tree_str)
    graph.write_png(file_name)

tree_graph(cart_final, X.columns, "cart_tree.png")

####################################
# 9. Extracting Decision Rules
####################################

tree_rules = export_text(cart_model, feature_names=list(X.columns))
print(tree_rules)

# Convert to Python code
print(skompile(cart_final.predict).to("python/code"))

# Convert to SQL
print(skompile(cart_final.predict).to("sqlalchemy/sqlite"))

# Convert to Excel
print(skompile(cart_final.predict).to("excel"))

#####################################
# 10. Prediction using Python Code
#####################################

def predict_with_rules(x):
    return (((((0 if x[0] <= 7.5 else 1) if x[5] <= 30.95 else 0 if x[6] <= 0.50 else 0)
              if x[5] <= 45.40 else 1 if x[2] <= 99.0 else 0)
             if x[7] <= 28.5 else (1 if x[5] <= 9.65 else 0) if x[5] <= 26.35 else (1 if x[1] <= 28.5 else 0)
             if x[1] <= 99.5 else 0 if x[6] <= 0.56 else 1)
            if x[1] <= 127.5 else 0)

X.columns

x = [6, 148, 70, 35, 0, 30, 0.62, 50]
print(predict_with_rules(x))

######################################
# 11. Saving and Loading Model
######################################

# Save model
joblib.dump(cart_final, "cart_final.pkl")

# Load model
cart_model_from_disc = joblib.load("cart_final.pkl")

# Predict using loaded model
x = pd.DataFrame([[6, 148, 70, 35, 0, 30, 0.62, 50]])
print(cart_model_from_disc.predict(x))
