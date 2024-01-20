import pandas as pd
import numpy as np
import os

DATA_PATH = os.path.join(os.path.join(os.path.dirname(__file__), 'data'), "training.csv")

DROP_COLS = [
    "number_of_stages",
    "frac_type",
    "bh_x","bh_y",
    "standardized_operator_name",
    "average_stage_length",
    "average_proppant_per_stage",
    "average_frac_fluid_per_stage",
    "average_proppant_per_stage",
    "average_frac_fluid_per_stage",
    "proppant_to_frac_fluid_ratio",
    "frac_fluid_to_proppant_ratio",
    "frac_seasoning",
    # adding after incorporating weak learner
    'horizontal_midpoint_x',
    'horizontal_midpoint_y',
    'horizontal_toe_x',
    'horizontal_toe_y',
    'surface_toe_len', 
    'toe_midpoint_len', 'toe_bh_len',
    'midpoint_bh_len', 'lateral_len',
    'surface_toe_len', 'toe_midpoint_len',
    'toe_bh_len', 'midpoint_bh_len',
    'lateral_len', 'custom_average_proppant',
    'proppant_intensity', 'frac_fluid_intensity',
    'pad_id'
]

RAW_DATA = pd.read_csv(DATA_PATH, index_col=0)

def preprocess_data(data):

    data.dropna(subset=["OilPeakRate"], inplace=True)

    data["surface_toe_len"] = np.sqrt((data.surface_x - data.horizontal_toe_x).to_numpy() ** 2 + (data.surface_y - data.horizontal_toe_y).to_numpy() ** 2)
    data["toe_midpoint_len"] = np.sqrt((data.horizontal_toe_x - data.horizontal_midpoint_x).to_numpy() ** 2 + (data.horizontal_toe_y - data.horizontal_midpoint_y).to_numpy() ** 2)
    data["toe_bh_len"] = np.sqrt((data.horizontal_toe_x - data.bh_x).to_numpy() ** 2 + (data.horizontal_toe_y - data.bh_y).to_numpy() ** 2)
    data["midpoint_bh_len"] = np.sqrt((data.horizontal_midpoint_x - data.bh_x).to_numpy() ** 2 + (data.horizontal_midpoint_y - data.bh_y).to_numpy() ** 2)

    data["lateral_len"] = data.toe_midpoint_len * 2 # proxy a number of stages de pipe

    data["custom_average_proppant"] = data.total_proppant / data.surface_toe_len

    data.drop(columns=DROP_COLS, inplace=True)

    data.to_csv(os.path.join(os.path.dirname(__file__), "data", "preprocessed.csv"), index=False)


if __name__ == "__main__":
    preprocess_data(RAW_DATA)

