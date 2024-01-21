import pandas as pd
import wandb
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, mean_absolute_error
from sklearn.model_selection import train_test_split


# read data
data = pd.read_csv("data/final.csv")

# split data
TARGET_COL = "OilPeakRate"
SEED = 1234

X = data.drop(columns=[TARGET_COL] + ["cluster"])
y = data[TARGET_COL]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

# train model