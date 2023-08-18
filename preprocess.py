#!/usr/bin/env python

# Libraries
import numpy as np
import pandas as pd

# import gc
# import re
# from sklearn.impute import KNNImputer
# from imblearn.over_sampling import SMOTE
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)


# One-hot encoding for categorical columns with get_dummies
def one_hot_encoder(df):
    df = pd.get_dummies(
        df,
        columns=[col for col in df.columns if df[col].dtype == "object"],
        dummy_na=True,
    )

    return df


# application_train/test
def application(df):
    df = df.reset_index(drop=True)
    # Remove applications with XNA CODE_GENDER
    df = df[df["CODE_GENDER"] != "XNA"]

    # NaN values for DAYS_EMPLOYED: 365.243 -> nan
    df["DAYS_EMPLOYED"].replace(365243, np.nan, inplace=True)

    # Categorical features with Binary encode (0 or 1; two categories)
    for bin_feature in ["CODE_GENDER", "FLAG_OWN_CAR", "FLAG_OWN_REALTY"]:
        df[bin_feature], uniques = pd.factorize(df[bin_feature])

    # Categorical features with One-Hot encode
    df = one_hot_encoder(df)

    return df