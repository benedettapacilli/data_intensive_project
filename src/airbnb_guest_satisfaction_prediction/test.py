# Airbnb Guest Satisfaction Prediction

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os.path
import kaggle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.kernel_ridge import KernelRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score


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


def plot_predictions(x_train, y_test, model):
    # Generate predicted values for test data
    y_pred = model.predict(x_train)

    for prediction, expected in zip(y_pred, y_train):
        print(f"predicted: {prediction:.2f}, expected: {expected:.2f}")


# Checking if the dataset Aemf1.csv is in the cwd
if not os.path.exists("Aemf1.csv"):
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(
        'dipeshkhemani/airbnb-cleaned-europe-dataset', path='.', unzip=True)

# Importing the dataset
dataset = pd.read_csv('Aemf1.csv')

# Checking for missing values
if dataset.isnull().values.any():
    print("Missing values found")
    # Removing missing values
    dataset = dataset.dropna()
    print("Missing values removed")
else:
    print("No missing values found")

print(dataset.describe())

# convert boolean values to int
dataset["Shared Room"] = dataset["Shared Room"].astype(int)
dataset["Private Room"] = dataset["Private Room"].astype(int)
dataset["Superhost"] = dataset["Superhost"].astype(int)

# create a folder to store the plots
if not os.path.exists("plots"):
    os.mkdir(path="plots")

for column in dataset.columns.drop("Guest Satisfaction"):
    dataset.plot.scatter(x=column, y="Guest Satisfaction").get_figure().savefig(
        "plots/" + column + ".png")

# Columns to be removed: Bedrooms, Business, City, Day, Multiple Rooms, Person Capacity, Private Room, Room Type, Shared Room, Normalised Attraction Index, Normalised Restraunt Index
dataset = dataset.drop(columns=["Bedrooms", "Business", "City", "Day", "Multiple Rooms", "Person Capacity",
                       "Private Room", "Room Type", "Shared Room", "Normalised Attraction Index", "Normalised Restraunt Index"])

# Remaining columns ['Price' 'Superhost' 'Cleanliness Rating' 'Guest Satisfaction' 'City Center (km)' 'Metro Distance (km' 'Attraction Index' 'Restraunt Index']
print("Remaining columns")
print(dataset.columns.values)

numerical_columns = ['Price', 'Cleanliness Rating',
                     'City Center (km)', 'Metro Distance (km)', 'Attraction Index', 'Restraunt Index']
binary_columns = ['Superhost']

# Separating training and test data with a 80:20 ratio in two different datasets
train_set, test_set = train_test_split(dataset, test_size=0.2, random_state=42)

x_train = train_set.drop(columns=["Guest Satisfaction"])
y_train = train_set["Guest Satisfaction"]
x_test = test_set.drop(columns=["Guest Satisfaction"])
y_test = test_set["Guest Satisfaction"]

# # Testing Linear Regression
# print("Linear Regression")
# model_poli_std_lin = Pipeline([
#     ('poly', PolynomialFeatures(degree=3, include_bias=False)),
#     ('scaler', StandardScaler()),
#     ('linear', LinearRegression())
# ], verbose=True)

# model_poli_std_lin.fit(x_train, y_train)

# print_eval(x_train, y_train, model_poli_std_lin)

# print_eval(x_test, y_test, model_poli_std_lin)

# # Testing Ridge Regression
# print("Ridge Regression")
# model_poli_std_rdg = Pipeline([
#     ('poly', PolynomialFeatures(degree=3, include_bias=False)),
#     ('scaler', StandardScaler()),
#     ('ridge', Ridge(alpha=1))
# ], verbose=True)

# model_poli_std_rdg.fit(x_train, y_train)

# print_eval(x_train, y_train, model_poli_std_rdg)

# print_eval(x_test, y_test, model_poli_std_rdg)

# # Testing Elastic Net Regression
# print("Elastic Net Regression")
# model_poli_std_en = Pipeline([
#     ('poly', PolynomialFeatures(degree=10, include_bias=False)),
#     ('scaler', StandardScaler()),
#     ('elasticnet', ElasticNet(alpha=0.1, l1_ratio=0.8))
# ], verbose=True)

# model_poli_std_en.fit(x_train, y_train)

# print_eval(x_train, y_train, model_poli_std_en)

# print_eval(x_test, y_test, model_poli_std_en)

# # Testing KernelRidge Regression
# print("KernelRidge Regression")
# model_poli_std_krr = Pipeline([
#     ('scaler', StandardScaler()),
#     ('krr', KernelRidge(alpha=1, kernel='poly', degree=10))
# ], verbose=True)

# model_poli_std_krr.fit(x_train, y_train)

# print_eval(x_train, y_train, model_poli_std_krr)

# print_eval(x_test, y_test, model_poli_std_krr)

# # Testing Regression trees
# print("Regression trees")

# model_poli_std_dtr = Pipeline([
#     ('preproc', ColumnTransformer(transformers=[
#         ('onehot', OneHotEncoder(sparse_output=False), binary_columns),
#         ('poly', PolynomialFeatures(degree=2, include_bias=False), numerical_columns),
#         ('scaler', StandardScaler(), numerical_columns),
#     ], remainder='passthrough', verbose=True)),
#     ('dtr', DecisionTreeRegressor(random_state=42))
# ], verbose=True)

# model_poli_std_dtr.fit(x_train, y_train)

# print_eval(x_train, y_train, model_poli_std_dtr)

# print_eval(x_test, y_test, model_poli_std_dtr)

# # Testing XGBoost Regressor
# print("XGBoost Regressor")

# model_poli_std_xgb = Pipeline([
#     ('preproc', ColumnTransformer(transformers=[
#         # ('onehot', OneHotEncoder(sparse_output=False), binary_columns),
#         # ('poly', PolynomialFeatures(degree=2, include_bias=False), numerical_columns),
#         ('scaler', StandardScaler(), numerical_columns),
#     ], remainder='passthrough', verbose=True)),
#     ('xgb', XGBRegressor(objective='reg:squarederror', reg_alpha=0.0,
#      reg_lambda=1.0, n_estimators=200, verbose_eval=True))
# ], verbose=True)

# model_poli_std_xgb.fit(x_train, y_train)

# print_eval(x_train, y_train, model_poli_std_xgb)

# print_eval(x_test, y_test, model_poli_std_xgb)

# # Testing Random Forest Regressor best model
# print("Random Forest Regressor")

# model_poli_std_rfr = Pipeline([
#     ('preproc', ColumnTransformer(transformers=[
#         ('scaler', StandardScaler(), numerical_columns),
#     ], remainder='passthrough', verbose=True)),
#     ('rfr', RandomForestRegressor(n_estimators=200, max_features="log2", n_jobs=-1))
# ], verbose=True)

# model_poli_std_rfr.fit(x_train, y_train)

# print_eval(x_train, y_train, model_poli_std_rfr)

# print_eval(x_test, y_test, model_poli_std_rfr)

# # Grid Search for Random Forest Regressor
# print("Grid Search for Random Forest Regressor")
# grid_params = {
#     'rfr__max_features': [None, 'sqrt', 'log2'],
#     'rfr__min_samples_split': [2, 5, 10],
#     'rfr__min_samples_leaf': [1, 2, 4],
#     'rfr__bootstrap': [True, False]
# }

# grid_search = GridSearchCV(
#     model_poli_std_rfr, grid_params, cv=5, n_jobs=-1)
# grid_search.fit(x_train, y_train)

# print(grid_search.best_params_)
# print(grid_search.best_score_)
# print(grid_search.best_estimator_)

# Testing Random Forest Regressor best model
print("Random Forest Regressor")

model_poli_std_rfr = Pipeline([
    ('preproc', ColumnTransformer(transformers=[
        ('scaler', StandardScaler(), numerical_columns),
    ], remainder='passthrough', verbose=True)),
    ('rfr', RandomForestRegressor(n_estimators=200,
     max_features="sqrt", bootstrap=False, n_jobs=-1))
], verbose=True)

model_poli_std_rfr.fit(x_train, y_train)

print_eval(x_train, y_train, model_poli_std_rfr)

print_eval(x_test, y_test, model_poli_std_rfr)

pd.DataFrame(model_poli_std_rfr.named_steps['rfr'].feature_importances_, index=x_train.columns).plot.bar(
    figsize=(20, 10), title="Feature Importance").get_figure().savefig("feature_importance.png")

# predict with these test data [70, 0, 8, 7, 2, 500, 500]
test = pd.DataFrame([[3000, 0, 3, 20, 2, 0, 0]], columns=['Price', 'Superhost', 'Cleanliness Rating',
                    'City Center (km)', 'Metro Distance (km)', 'Attraction Index', 'Restraunt Index'])
print(model_poli_std_rfr.predict(test))
