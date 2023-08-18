import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer

# from sklearn.impute import IterativeImputer
from sklearn.impute import KNNImputer
from preprocessing import preprocess
from imblearn.over_sampling import SMOTE
import argparse


def gen_train_valid_test(smote=False, impute=False):
    # Read and preprocess DFs
    app_train_df = pd.read_csv("input/application_train.csv")
    app_test_df = pd.read_csv("input/application_test.csv")
    preproc_app_train_df = preprocess(app_train_df)
    preproc_app_test_df = preprocess(app_test_df)
    preproc_app_train_cols = preproc_app_train_df.columns.to_list()
    preproc_app_test_cols = preproc_app_test_df.columns.to_list()

    # Filter out rows with null values in the "TARGET" column
    app_train_df = app_train_df[app_train_df["TARGET"].notnull()].reset_index(drop=True)

    # Ensure column parity between train and test
    for col_name in preproc_app_train_cols:
        if col_name != "TARGET" and col_name not in preproc_app_test_cols:
            if pd.api.types.is_bool_dtype(preproc_app_train_df[col_name]):
                preproc_app_test_df[col_name] = False
            else:
                preproc_app_test_df[col_name] = np.nan

    preproc_app_test_df = preproc_app_test_df[
        [x for x in preproc_app_train_cols if x != "TARGET"]
    ].reset_index(drop=True)

    # Convert boolean values to integers
    bool_cols = [
        x for x in preproc_app_train_cols if preproc_app_train_df[x].dtype == "bool"
    ]
    preproc_app_train_df[bool_cols] = preproc_app_train_df[bool_cols].astype(int)

    # Smote does not accept missing values; they must be imputed
    if smote or impute:
        # # Create and fit the iterative imputer
        # imputed_data = IterativeImputer(random_state=0).fit_transform(
        #     preproc_app_train_df
        # )
        # Create and fit the knn imputer
        imputed_data = KNNImputer(n_neighbors=5).fit_transform(preproc_app_train_df)

        # Convert the imputed data back to a DataFrame
        preproc_app_train_df = pd.DataFrame(
            imputed_data,
            columns=preproc_app_train_df.columns,
            index=preproc_app_train_df.index,
        )

        # Round the imputed values for boolean columns
        preproc_app_train_df[bool_cols] = (
            preproc_app_train_df[bool_cols].round().astype(int)
        )

    # Apply SMOTE if the parameter is True
    if smote:
        smote = SMOTE(random_state=0)
        X = preproc_app_train_df.drop(columns=["TARGET"])
        y = preproc_app_train_df["TARGET"]
        X_resampled, y_resampled = smote.fit_resample(X, y)
        preproc_app_train_df = pd.concat([X_resampled, y_resampled], axis=1)

    # Add a stratification column with the same values as the target variable
    preproc_app_train_df["stratify_col"] = preproc_app_train_df["TARGET"]

    # Perform a stratified train/test split using the stratify_col column
    preproc_app_train_df, preproc_app_valid_df = train_test_split(
        preproc_app_train_df,
        test_size=0.2,
        random_state=0,
        stratify=preproc_app_train_df["stratify_col"],
    )

    # Drop the stratification column from both DataFrames
    preproc_app_train_df = preproc_app_train_df.drop(
        columns=["stratify_col"]
    ).reset_index(drop=True)
    preproc_app_valid_df = preproc_app_valid_df.drop(
        columns=["stratify_col"]
    ).reset_index(drop=True)

    return [preproc_app_train_df, preproc_app_valid_df, preproc_app_test_df]


def main(smote, impute):
    (
        preproc_app_train_df,
        preproc_app_valid_df,
        preproc_app_test_df,
    ) = gen_train_valid_test(smote=smote, impute=impute)
    # Save DFs
    addit_step = "_smote" if smote else "_imputed" if impute else ""
    print(f"Preprocessed Training DataFrame shape: {preproc_app_train_df.shape}")
    preproc_app_train_df.to_feather(f"preproc{addit_step}_app_train_df.ftr")
    print(f"Preprocessed Validation DataFrame shape: {preproc_app_valid_df.shape}")
    preproc_app_valid_df.to_feather(f"preproc{addit_step}_app_valid_df.ftr")
    print(f"Preprocessed Testing DataFrame shape: {preproc_app_test_df.shape}")
    preproc_app_test_df.to_feather(f"preproc{addit_step}_app_test_df.ftr")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smote", action="store_true", default=False, help="Enable SMOTE if specified"
    )
    parser.add_argument(
        "--impute",
        action="store_true",
        default=False,
        help="Enable imputation if specified",
    )

    args = parser.parse_args()

    main(args.smote, args.impute)
