import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

DATA_PATH = os.path.join(os.path.join(os.path.dirname(__file__), 'data'), "training.csv")

DROP_COLS = [
    "frac_type",
    "bh_x","bh_y",
    "standardized_operator_name",
    "average_proppant_per_stage",
    "average_frac_fluid_per_stage",
    "average_proppant_per_stage",
    "average_frac_fluid_per_stage",
    "proppant_to_frac_fluid_ratio",
    "frac_fluid_to_proppant_ratio",
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
    # Compute the log of frac_seasoning to make it less skewed
    data["frac_seasoning"] = np.log(data["frac_seasoning"] + 1)

    data["custom_average_proppant"] = data.total_proppant / data.surface_toe_len

    # Estimate the total and lateral number of stages with a linear regressor
    data = data\
           .merge(estimate_stages(data, "total"), left_index=True, right_index=True)\
           .merge(estimate_stages(data, "lateral"), left_index=True, right_index=True)
    
    # overwrite computed number of stages in the perforation
    data["number_of_stages"].fillna(data["predicted_number_of_total_stages"], inplace=True)
    data.drop(columns=["predicted_number_of_total_stages"], inplace=True)
    
    # overwrite computed number of stages in the lateral
    # data.drop(columns=["average_stage_length"], inplace=True) # we don't need this anymore
    data["number_of_lateral_stages"] = data["bin_lateral_length"] * data["average_stage_length"]
    data["number_of_lateral_stages"].fillna(data["predicted_number_of_lateral_stages"], inplace=True)
    data.drop(columns=["predicted_number_of_lateral_stages", "average_stage_length"], inplace=True)

    data["sibling_well"] = (data["well_family_relationship"] == "Sibling Well").astype(int)
    data["batch_concurrent_frac"] = (data['batch_frac_classification'] == "Batch-Concurrent Frac").astype(int)

    data.drop(columns=DROP_COLS, inplace=True)

    data.to_csv(os.path.join(os.path.dirname(__file__), "data", "preprocessed.csv"), index=False)

    return data

def get_custom_estimations(data):
    # Estimate the total and lateral number of stages with a linear regressor
    data = data\
           .merge(estimate_stages(data, "total"), left_index=True, right_index=True)\
           .merge(estimate_stages(data, "lateral"), left_index=True, right_index=True)
    
    # overwrite computed number of stages in the perforation
    data["number_of_stages"].fillna(data["predicted_number_of_total_stages"], inplace=True)
    
    # overwrite computed number of stages in the lateral
    # data.drop(columns=["average_stage_length"], inplace=True) # we don't need this anymore
    data["number_of_lateral_stages"] = data["bin_lateral_length"] * data["average_stage_length"]
    data["number_of_lateral_stages"].fillna(data["predicted_number_of_lateral_stages"], inplace=True)

    return data

def estimate_stages(data_original, stages_to_estimate):
    data = data_original.copy()

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ])

    if stages_to_estimate == "total":
        predictive_cols = [
            "total_proppant",
            "total_fluid",
            "bin_lateral_length",
            "frac_seasoning",
            "gross_perforated_length"
        ]
        target_cols = ["number_of_stages"]
    elif stages_to_estimate == "lateral":
        predictive_cols = [
            "total_proppant",
            "total_fluid",
            # "frac_seasoning",
            # "true_vertical_depth",
            "gross_perforated_length"
        ]
        target_cols = ["bin_lateral_length", "average_stage_length"]
        
    data = data[predictive_cols + target_cols]

    # Impute NaNs in all columns except the ones that will be targets
    for col in predictive_cols:
        if col not in target_cols:
            data[col] = data[col].fillna(data[col].mean())

    # Now filter where the targets are NaN, since we can't use them
    data = data.dropna(subset=target_cols)

    if stages_to_estimate == "total":
        y = data["number_of_stages"]
        X = data.drop(["number_of_stages"], axis=1)
    elif stages_to_estimate == "lateral":
        y = data["average_stage_length"] * data["bin_lateral_length"]
        X = data.drop(["average_stage_length", "bin_lateral_length"], axis=1)

    pipeline.fit(X, y)

    X_test = data_original[predictive_cols].dropna()
    y_test = pd.Series(pipeline.predict(X_test), index=X_test.index)
    y_test = y_test.rename(f"predicted_number_of_{stages_to_estimate}_stages")

    return y_test

def preprocess_modeling_data(preprocessed, ohe=True, mean_imputation=True):
    """
    Prepare dataset for modeling
    """

    # cluster_features = pd.read_csv("data/cluster_label_features.csv")
    # cluster_features.rename(columns={"Unnamed: 0": "original_index"}, inplace=True)

    # cluster_features = cluster_features[["original_index", "cluster", "opr_mean_pct_diff"]].copy()

    # preprocessed = preprocessed.merge(cluster_features, right_on="original_index", left_index=True)
    # preprocessed.drop(columns=["original_index"], inplace=True)

    # pri

    if mean_imputation:
        preprocessed["gross_perforated_length"] = preprocessed["gross_perforated_length"].fillna(preprocessed["gross_perforated_length"].mean())
        preprocessed["total_proppant"] = preprocessed["total_proppant"].fillna(preprocessed["total_proppant"].mean())
        preprocessed["total_fluid"] = preprocessed["total_fluid"].fillna(preprocessed["total_fluid"].mean())
        preprocessed["bin_lateral_length"] = preprocessed["bin_lateral_length"].fillna(preprocessed["bin_lateral_length"].mean())
        preprocessed["true_vertical_depth"] = preprocessed["true_vertical_depth"].fillna(preprocessed["true_vertical_depth"].mean())

    preprocessed.ffs_frac_type = preprocessed.ffs_frac_type.fillna("Undefined").map({
                                                        "Slickwater Crosslink Hybrid": "sch",
                                                        "Slickwater Only": "so",
                                                        "8810.144334222554": "number",
                                                        "Slickwater Linear Hybrid": "slh",
                                                        "Undefined": "undefined",
                                                        "Crosslink Only": "co",
                                                        "Linear Only": "lo",
                                                    })
    preprocessed.ffs_frac_type.value_counts(dropna=False)

    if ohe:
        # One Hot Encode data
        ohe_cols = ["ffs_frac_type",
        "relative_well_position",
        "batch_frac_classification",
        "well_family_relationship"]

        for col in ohe_cols:
            preprocessed = pd.concat([preprocessed, pd.get_dummies(preprocessed[col], prefix=col)], axis=1)
            preprocessed.drop(columns=[col], inplace=True)

    # preprocessed.drop(columns=["cluster"], inplace=True)
    
    return preprocessed


def prepare_test_data(data):

    data["surface_toe_len"] = np.sqrt((data.surface_x - data.horizontal_toe_x).to_numpy() ** 2 + (data.surface_y - data.horizontal_toe_y).to_numpy() ** 2)
    data["toe_midpoint_len"] = np.sqrt((data.horizontal_toe_x - data.horizontal_midpoint_x).to_numpy() ** 2 + (data.horizontal_toe_y - data.horizontal_midpoint_y).to_numpy() ** 2)
    data["toe_bh_len"] = np.sqrt((data.horizontal_toe_x - data.bh_x).to_numpy() ** 2 + (data.horizontal_toe_y - data.bh_y).to_numpy() ** 2)
    data["midpoint_bh_len"] = np.sqrt((data.horizontal_midpoint_x - data.bh_x).to_numpy() ** 2 + (data.horizontal_midpoint_y - data.bh_y).to_numpy() ** 2)

    data["lateral_len"] = data.toe_midpoint_len * 2 # proxy a number of stages de pipe
    # Compute the log of frac_seasoning to make it less skewed
    data["frac_seasoning"] = np.log(data["frac_seasoning"] + 1)

    data["custom_average_proppant"] = data.total_proppant / data.surface_toe_len

    data["sibling_well"] = (data["well_family_relationship"] == "Sibling Well").astype(int)
    data["batch_concurrent_frac"] = (data['batch_frac_classification'] == "Batch-Concurrent Frac").astype(int)

    data.drop(columns=[
        "ffs_frac_type",
        "relative_well_position",
        "batch_frac_classification",
        "well_family_relationship",
        "bin_lateral_length",
        "frac_type",
        "bh_x","bh_y",
        "standardized_operator_name",
        "average_proppant_per_stage",
        "average_frac_fluid_per_stage",
        "average_proppant_per_stage",
        "average_frac_fluid_per_stage",
        "proppant_to_frac_fluid_ratio",
        "frac_fluid_to_proppant_ratio",
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
    ], inplace=True)

    return data


if __name__ == "__main__":
    preprocessed = preprocess_data(RAW_DATA)
    print("Preprocessed data shape:")
    print(preprocessed.shape)

    final = preprocess_modeling_data(preprocessed, ohe=False)

    final.drop(columns=["ffs_frac_type",
        "relative_well_position",
        "batch_frac_classification",
        "well_family_relationship",
        "bin_lateral_length",
        "number_of_lateral_stages"
        ], inplace=True)

    print("Final data shape:")
    print(final.shape)
    final.to_csv("data/final.csv", index=False)

