# Airbnb Guest Satisfaction Prediction

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os.path
import kaggle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, Lasso
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

# Columns to be removed: Bedrooms, Business, City, Day, Multiple Rooms, Attraction Index, Restraunt Index, Person Capacity, Private Room, Room Type, Shared Room, Normalised Attraction Index, Normalised Restraunt Index
dataset = dataset.drop(columns=["Bedrooms", "Business", "City", "Day", "Multiple Rooms", "Attraction Index", "Restraunt Index",
                       "Person Capacity", "Private Room", "Room Type", "Shared Room", "Normalised Attraction Index", "Normalised Restraunt Index"])

# Separating training and test data with a 80:20 ratio in two different datasets
train_set, test_set = train_test_split(dataset, test_size=0.2, random_state=42)

x_train = train_set.drop(columns=["Guest Satisfaction"])
y_train = train_set["Guest Satisfaction"]
x_test = test_set.drop(columns=["Guest Satisfaction"])
y_test = test_set["Guest Satisfaction"]

# Testing Ridge Regression
print("Ridge Regression")
model = Pipeline([
    ('poly', PolynomialFeatures(degree=10, include_bias=False)),
    ('scaler', StandardScaler()),
    ('ridge', Ridge(alpha=1))
])

model.fit(x_train, y_train)

print_eval(x_train, y_train, model)

print_eval(x_test, y_test, model)

# plot_predictions(x_train, y_train, model)
# pd.Series(model.named_steps["ridge"].coef_, x_train.columns)

# Testing Lasso Regression
print("Lasso Regression")
model = Pipeline([
    ('scaler', StandardScaler()),
    ('lasso', Lasso(alpha=0.1))
])

model.fit(x_train, y_train)

print_eval(x_train, y_train, model)

# plot_predictions(x_train, y_train, model)
pd.Series(model.named_steps["lasso"].coef_, x_train.columns)
