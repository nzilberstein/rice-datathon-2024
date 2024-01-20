import pandas as pd
import numpy as np
import os

DATA_PATH = os.path.join(os.path.join(os.path.dirname(__file__), 'data'), "training.csv")

DROP_COLS = [
    "number_of_stages",
    "frac_type",
]

RAW_DATA = pd.read_csv(DATA_PATH, index_col=0)

def preprocess_data(data):

    data.dropna(subset=["OilPeakRate"], inplace=True)

    data["surface_toe_len"] = np.sqrt((data.surface_x - data.horizontal_toe_x).to_numpy() ** 2 + (data.surface_y - data.horizontal_toe_y).to_numpy() ** 2)
    data["surface_midpoint_len"] = np.sqrt((data.surface_x - data.horizontal_midpoint_x).to_numpy() ** 2 + (data.surface_y - data.horizontal_midpoint_y).to_numpy() ** 2)
    data["surface_bh_len"] = np.sqrt((data.surface_x - data.bh_x).to_numpy() ** 2 + (data.surface_y - data.bh_y).to_numpy() ** 2)
    data["toe_midpoint_len"] = np.sqrt((data.horizontal_toe_x - data.horizontal_midpoint_x).to_numpy() ** 2 + (data.horizontal_toe_y - data.horizontal_midpoint_y).to_numpy() ** 2)
    data["toe_bh_len"] = np.sqrt((data.horizontal_toe_x - data.bh_x).to_numpy() ** 2 + (data.horizontal_toe_y - data.bh_y).to_numpy() ** 2)
    data["midpoint_bh_len"] = np.sqrt((data.horizontal_midpoint_x - data.bh_x).to_numpy() ** 2 + (data.horizontal_midpoint_y - data.bh_y).to_numpy() ** 2)


    data["custom_average_proppant"] = data.total_proppant / data.surface_toe_len

    data.to_csv(os.path.join(os.path.dirname(__file__), "data", "preprocessed.csv"), index=False)


if __name__ == "__main__":
    preprocess_data(RAW_DATA)

