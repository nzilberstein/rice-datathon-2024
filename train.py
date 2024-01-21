import numpy as np
import pandas as pd
import wandb
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from wandb.xgboost import WandbCallback
from sklearn.preprocessing import StandardScaler


# read data
data = pd.read_csv("data/final.csv")

# drop outlier
data.drop(12936, inplace=True)
# data.drop(columns=["cluster"])


print(f"Working with columns: {data.columns}")
print(f"Data shape: {data.shape}")

# split data
TARGET_COL = "OilPeakRate"
SEED = 1234

X = data.drop(columns=[TARGET_COL])
y = data[TARGET_COL]

scaler = StandardScaler()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


def train_model():
    wandb.init()

    model = xgb.XGBRegressor(
        max_depth=wandb.config.max_depth,
        learning_rate=wandb.config.learning_rate,
        n_estimators=wandb.config.n_estimators,
        gamma=wandb.config.gamma,
        min_child_weight=wandb.config.min_child_weight,
        subsample=wandb.config.subsample,
        alpha=wandb.config.alpha,
        reg_lambda=wandb.config.reg_lambda,
        colsample_bytree=wandb.config.colsample_bytree,
        objective="reg:squarederror",
        booster="gbtree"
    )
    model.fit(X_train, y_train)

    # Predict on test set
    y_preds = model.predict(X_test)
    y_pred_train = model.predict(X_train)

    # Evaluate predictions
    rmse_test = np.sqrt(mean_squared_error(y_test, y_preds))
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    train_test_diff = rmse_test - rmse_train

    # Log model performance metrics to W&B
    wandb.log({"rmse_test": rmse_test, "rmse_train": rmse_train, "train_test_diff": train_test_diff})


sweep_configs = {
    "method": "random",
    "metric": {"name": "rmse_train", "goal": "minimize"},
    "parameters": {
        "max_depth": {"values": [1, 2, 5, 10]},
        "learning_rate": {"distribution": "uniform", "min": 0, "max": 0.1},
        "n_estimators": {"values": [100, 500, 1000]},
        "gamma": {"distribution": "uniform", "min": 1e-2, "max": 100},
        "min_child_weight": {"distribution": "uniform", "min": 1, "max": 100},
        "subsample": {"values": [0.3, 0.4, 0.5]},
        "n_estimators": {"values": [100, 200, 300]},
        "alpha": {"distribution": "uniform", "min": 1e-2, "max": 1},
        "reg_lambda": {"distribution": "uniform", "min": 1e-2, "max": 1},
        "colsample_bytree": {"values": [0.5, .7, .9]},
    },
}
# train model

sweep_id = wandb.sweep(sweep_configs, project="rice-datathon-2024")
wandb.agent(sweep_id=sweep_id, function=train_model)





