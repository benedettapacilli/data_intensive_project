# Diabetes Prediction

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os.path
import kaggle
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


def relative_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true))


def print_eval(X, y, model):
    preds = model.predict(X)
    mse = mean_squared_error(y, preds)
    re = relative_error(y, preds)
    r2 = r2_score(y, preds)
    print(f"   Mean squared error: {mse:.5}")
    print(f"       Relative error: {re:.5%}")
    print(f"R-squared coefficient: {r2:.5}")


def separator_2d(model, x1):
    # ricaviamo w e b dal modello
    w = model.coef_[0]
    b = model.intercept_[0]
    # riportiamo in NumPy la formula sopra
    return -x1 * w[0] / w[1] - b / w[1]


def plot_separator_on_data(X, y, model=None):
    X = np.array(X)
    colors = pd.Series(y).map({0: "blue", 1: "red"})
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 2], X[:, 3], c=colors)
    if model is not None:
        xlim, ylim = plt.xlim(), plt.ylim()
        sep_x = np.linspace(*xlim, 2)
        sep_y = separator_2d(model, sep_x)
        plt.plot(sep_x, sep_y, c="green", linewidth=2)
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.show()


# Checking if the dataset diabetes_prediction_dataset.csv is in the cwd
if not os.path.exists("diabetes_prediction_dataset.csv"):
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(
        'iammustafatz/diabetes-prediction-dataset', path='.', unzip=True)

# Importing the dataset
dataset = pd.read_csv('diabetes_prediction_dataset.csv')

# Checking for missing values
if dataset.isnull().values.any():
    print("Missing values found")
    # Removing missing values
    dataset = dataset.dropna()
    print("Missing values removed")
else:
    print("No missing values found")

print(dataset.describe())

print("Remaining columns:")
print(dataset.columns.values)

numerical_columns = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']

binary_columns = ['hypertension', 'heart_disease']

categorical_columns = ['gender', 'smoking_history']

# Separating training and test data (80% - 20%)
train_set, test_set = train_test_split(dataset, test_size=0.2, random_state=42)

x_train = train_set.drop(columns=['diabetes'])
y_train = train_set['diabetes']
x_test = test_set.drop(columns=['diabetes'])
y_test = test_set['diabetes']

# # Testing Logistic Regression
# print("Logistic Regression")

# model_lr = Pipeline([
#     ('preproc', ColumnTransformer(transformers=[
#         ('scaler', StandardScaler(), numerical_columns),
#         ('onehot', OneHotEncoder(), categorical_columns),
#         ('passthrough', 'passthrough', binary_columns)
#     ], verbose=True)),
#     ('model', LogisticRegression(C=0.1, solver='saga', penalty='l1'))
# ])

# model_lr.fit(x_train, y_train)

# print("Training set:")
# print_eval(x_train, y_train, model_lr)

# print("Test set:")
# print_eval(x_test, y_test, model_lr)

# scaler = StandardScaler()
# onehot = OneHotEncoder()
# x = model_lr['preproc'].transform(x_train)
# y = y_train

# plot_separator_on_data(x, y, model_lr['model'])

# # Testing Different Classifiers with Grid Search
# grid = {
#     'model': [LogisticRegression(C=0.1, solver='saga', penalty='l1'), DecisionTreeClassifier(), RandomForestClassifier()],
# }

# grid_search = GridSearchCV(model_lr, grid, cv=5, verbose=3, n_jobs=-1)

# grid_search.fit(x_train, y_train)

# print(grid_search.best_params_)

# Testing Random Forest Classifier with Grid Search
print("Random Forest Classifier")

model_rfc = Pipeline([
    ('preproc', ColumnTransformer(transformers=[
        ('scaler', StandardScaler(), numerical_columns),
        ('onehot', OneHotEncoder(), categorical_columns),
        ('passthrough', 'passthrough', binary_columns)
    ], verbose=True)),
    ('model', RandomForestClassifier(n_jobs=-1, bootstrap=True, criterion='gini', max_features='sqrt', n_estimators=300))
])

grid = {
    'model__max_depth': [80, 90, 100, 110],
    'model__min_samples_leaf': [1, 2, 4],
    'model__min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(model_rfc, grid, cv=5, verbose=3, n_jobs=-1)

grid_search.fit(x_train, y_train)

print(grid_search.best_params_)
