#!/usr/bin/env python

# Libraries
import numpy as np
import pandas as pd
import gc
import re
from sklearn.impute import KNNImputer
from imblearn.over_sampling import SMOTE
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)


# One-hot encoding for categorical columns with get_dummies
def one_hot_encoder(df, nan_as_category=True):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == "object"]
    df = pd.get_dummies(df, columns=categorical_columns, dummy_na=nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]

    return df, new_columns


def remove_prefix(df, prefix):
    df.columns = df.columns.str.replace(f"^{prefix}", "", regex=True)

    return df


# application_train/test
def application(df, nan_as_category=False):
    df = df.reset_index(drop=True)
    df = pd.read_csv("input/application_train.csv")
    # Optional: Remove 4 applications with XNA CODE_GENDER (train set)
    df = df[df["CODE_GENDER"] != "XNA"]

    # Categorical features with Binary encode (0 or 1; two categories)
    for bin_feature in ["CODE_GENDER", "FLAG_OWN_CAR", "FLAG_OWN_REALTY"]:
        df[bin_feature], uniques = pd.factorize(df[bin_feature])

    # Categorical features with One-Hot encode
    df, cat_cols = one_hot_encoder(df, nan_as_category)

    # NaN values for DAYS_EMPLOYED: 365.243 -> nan
    df["DAYS_EMPLOYED"].replace(365243, np.nan, inplace=True)

    # Add prefix to all columns except 'SK_ID_CURR'
    df.columns = [
        "APPL_" + col if col not in ["SK_ID_CURR", "TARGET"] else col
        for col in df.columns.to_list()
    ]

    return df, df.columns.to_list()


def appl_engin_feats(df, appl_cols):
    df = df[appl_cols]
    df = remove_prefix(df, "APPL_")
    # Some simple new features (percentages)
    df["DAYS_EMPLOYED_PERC"] = df["DAYS_EMPLOYED"] / df["DAYS_BIRTH"]
    df["INCOME_CREDIT_PERC"] = df["AMT_INCOME_TOTAL"] / df["AMT_CREDIT"]
    df["INCOME_PER_PERSON"] = df["AMT_INCOME_TOTAL"] / df["CNT_FAM_MEMBERS"]
    df["ANNUITY_INCOME_PERC"] = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"]
    df["PAYMENT_RATE"] = df["AMT_ANNUITY"] / df["AMT_CREDIT"]

    return df


# bureau & bureau_balance
def bureau_and_balance(nan_as_category=True):
    bureau = pd.read_csv("input/bureau.csv")
    bb = pd.read_csv("input/bureau_balance.csv")
    bureau.columns = [
        "BUREAU_" + col if col not in ["SK_ID_CURR", "SK_ID_BUREAU"] else col
        for col in bureau.columns.to_list()
    ]
    bb.columns = [
        "BB_" + col if col != "SK_ID_BUREAU" else col for col in bb.columns.to_list()
    ]
    # bb, bb_cat = one_hot_encoder(bb, nan_as_category)
    # bureau, bureau_cat = one_hot_encoder(bureau, nan_as_category)
    bureau_cols = bureau.columns.to_list()
    bureau = bureau.join(bb, how="left", on="SK_ID_BUREAU")

    # return bureau, bureau_cols, bb.columns.to_list(), bureau_cat, bb_cat
    return bureau, bureau_cols, bb.columns.to_list()


def bureau_engin_feats(df, bureau_cols, bb_cols, bureau_cat, bb_cat):
    bb = df[bb_cols]
    bureau = df[bureau_cols]

    bb = remove_prefix(df, "BB_")
    bureau = remove_prefix(df, "BUREAU_")

    # Bureau balance: Perform aggregations and merge with bureau.csv
    bb_aggregations = {"MONTHS_BALANCE": ["min", "max", "size"]}

    for col in bb_cat:
        bb_aggregations[col] = ["mean"]

    bb_agg = bb.groupby("SK_ID_BUREAU").agg(bb_aggregations)
    bb_agg.columns = pd.Index(
        [e[0] + "_" + e[1].upper() for e in bb_agg.columns.tolist()]
    )
    bureau = bureau.join(bb_agg, how="left", on="SK_ID_BUREAU")

    bureau.drop(["SK_ID_BUREAU"], axis=1, inplace=True)
    del bb, bb_agg
    gc.collect()

    # Bureau and bureau_balance numeric features
    num_aggregations = {
        "DAYS_CREDIT": ["min", "max", "mean", "var"],
        "DAYS_CREDIT_ENDDATE": ["min", "max", "mean"],
        "DAYS_CREDIT_UPDATE": ["mean"],
        "CREDIT_DAY_OVERDUE": ["max", "mean"],
        "AMT_CREDIT_MAX_OVERDUE": ["mean"],
        "AMT_CREDIT_SUM": ["max", "mean", "sum"],
        "AMT_CREDIT_SUM_DEBT": ["max", "mean", "sum"],
        "AMT_CREDIT_SUM_OVERDUE": ["mean"],
        "AMT_CREDIT_SUM_LIMIT": ["mean", "sum"],
        "AMT_ANNUITY": ["max", "mean"],
        "CNT_CREDIT_PROLONG": ["sum"],
        "MONTHS_BALANCE_MIN": ["min"],
        "MONTHS_BALANCE_MAX": ["max"],
        "MONTHS_BALANCE_SIZE": ["mean", "sum"],
    }
    # Bureau and bureau_balance categorical features
    cat_aggregations = {}

    for cat in bureau_cat:
        cat_aggregations[cat] = ["mean"]

    for cat in bb_cat:
        cat_aggregations[cat + "_MEAN"] = ["mean"]

    bureau_agg = bureau.groupby("SK_ID_CURR").agg(
        {**num_aggregations, **cat_aggregations}
    )
    bureau_agg.columns = pd.Index(
        ["BURO_" + e[0] + "_" + e[1].upper() for e in bureau_agg.columns.tolist()]
    )
    # Bureau: Active credits - using only numerical aggregations
    active = bureau[bureau["CREDIT_ACTIVE_Active"] == 1]
    active_agg = active.groupby("SK_ID_CURR").agg(num_aggregations)
    active_agg.columns = pd.Index(
        ["ACTIVE_" + e[0] + "_" + e[1].upper() for e in active_agg.columns.tolist()]
    )
    bureau_agg = bureau_agg.join(active_agg, how="left", on="SK_ID_CURR")

    del active, active_agg
    gc.collect()

    # Bureau: Closed credits - using only numerical aggregations
    closed = bureau[bureau["CREDIT_ACTIVE_Closed"] == 1]
    closed_agg = closed.groupby("SK_ID_CURR").agg(num_aggregations)
    closed_agg.columns = pd.Index(
        ["CLOSED_" + e[0] + "_" + e[1].upper() for e in closed_agg.columns.tolist()]
    )
    bureau_agg = bureau_agg.join(closed_agg, how="left", on="SK_ID_CURR")

    del closed, closed_agg, bureau
    gc.collect()

    return bureau_agg


# installments_payments
def installments_payments(nan_as_category=True):
    ins = pd.read_csv("input/installments_payments.csv")
    ins.columns = [
        # "INSTL_" + col if col not in ["SK_ID_PREV", "SK_ID_CURR"] else col
        "INSTL_" + col if col != "SK_ID_CURR" else col
        for col in ins.columns
    ]
    # ins, ins_cat = one_hot_encoder(ins, nan_as_category)

    # return ins, ins.columns.to_list(), ins_cat
    return ins, ins.columns.to_list()


def installments_engin_feats(df, ins_cols, ins_cat):
    ins = df[ins_cols]
    ins = remove_prefix(ins, "INSTL_")
    # Percentage and difference paid in each installment (amount paid and installment value)
    ins["PAYMENT_PERC"] = ins["AMT_PAYMENT"] / ins["AMT_INSTALMENT"]
    ins["PAYMENT_DIFF"] = ins["AMT_INSTALMENT"] - ins["AMT_PAYMENT"]
    # Days past due and days before due (no negative values)
    ins["DPD"] = ins["DAYS_ENTRY_PAYMENT"] - ins["DAYS_INSTALMENT"]
    ins["DBD"] = ins["DAYS_INSTALMENT"] - ins["DAYS_ENTRY_PAYMENT"]
    ins["DPD"] = ins["DPD"].apply(lambda x: x if x > 0 else 0)
    ins["DBD"] = ins["DBD"].apply(lambda x: x if x > 0 else 0)
    # Features: Perform aggregations
    aggregations = {
        "NUM_INSTALMENT_VERSION": ["nunique"],
        "DPD": ["max", "mean", "sum"],
        "DBD": ["max", "mean", "sum"],
        "PAYMENT_PERC": ["max", "mean", "sum", "var"],
        "PAYMENT_DIFF": ["max", "mean", "sum", "var"],
        "AMT_INSTALMENT": ["max", "mean", "sum"],
        "AMT_PAYMENT": ["min", "max", "mean", "sum"],
        "DAYS_ENTRY_PAYMENT": ["max", "mean", "sum"],
    }

    for cat in ins_cat:
        aggregations[cat] = ["mean"]

    ins_agg = ins.groupby("SK_ID_CURR").agg(aggregations)
    ins_agg.columns = pd.Index(
        ["INSTAL_" + e[0] + "_" + e[1].upper() for e in ins_agg.columns.tolist()]
    )
    # Count installments accounts
    ins_agg["INSTAL_COUNT"] = ins.groupby("SK_ID_CURR").size()

    del ins
    gc.collect()

    return ins_agg


# pos_cash_balance
def pos_cash(nan_as_category=True):
    pos = pd.read_csv("input/POS_CASH_balance.csv")
    pos.columns = [
        # "POS_" + col if col not in ["SK_ID_PREV", "SK_ID_CURR"] else col
        "POS_" + col if col != "SK_ID_CURR" else col
        for col in pos.columns
    ]
    # pos, pos_cat = one_hot_encoder(pos, nan_as_category)

    # return pos, pos.columns.to_list(), pos_cat
    return pos, pos.columns.to_list(), pos_cat


def pos_engin_feats(df, pos_cols, pos_cat):
    pos = df[pos_cols]
    pos = remove_prefix(pos, "POS_")
    # Features
    aggregations = {
        "MONTHS_BALANCE": ["max", "mean", "size"],
        "SK_DPD": ["max", "mean"],
        "SK_DPD_DEF": ["max", "mean"],
    }

    for cat in pos_cat:
        aggregations[cat] = ["mean"]

    pos_agg = pos.groupby("SK_ID_CURR").agg(aggregations)
    pos_agg.columns = pd.Index(
        ["POS_" + e[0] + "_" + e[1].upper() for e in pos_agg.columns.tolist()]
    )
    # Count pos cash accounts
    pos_agg["POS_COUNT"] = pos.groupby("SK_ID_CURR").size()

    del pos
    gc.collect()

    return pos_agg


# credit_card_balance
def credit_card_balance(nan_as_category=True):
    cc = pd.read_csv("input/credit_card_balance.csv")
    cc.columns = [
        "CC_" + col if col not in ["SK_ID_PREV", "SK_ID_CURR"] else col
        for col in cc.columns
    ]
    cc = one_hot_encoder(cc, nan_as_category)[0]

    return cc, cc.columns.to_list()


def credit_card_balance_engin_feats(df, cc_cols):
    cc = df[cc_cols]
    cc = remove_prefix(cc, "CC_")

    # General aggregations
    cc.drop(["SK_ID_PREV"], axis=1, inplace=True)

    cc_agg = cc.groupby("SK_ID_CURR").agg(["min", "max", "mean", "sum", "var"])
    cc_agg.columns = pd.Index(
        ["CC_" + e[0] + "_" + e[1].upper() for e in cc_agg.columns.tolist()]
    )
    # Count credit card lines
    cc_agg["CC_COUNT"] = cc.groupby("SK_ID_CURR").size()

    del cc
    gc.collect()

    return cc_agg


# previous_applications
def previous_applications(nan_as_category=True):
    prev = pd.read_csv("input/previous_application.csv")

    # Days 365.243 values -> nan
    prev["DAYS_FIRST_DRAWING"].replace(365243, np.nan, inplace=True)
    prev["DAYS_FIRST_DUE"].replace(365243, np.nan, inplace=True)
    prev["DAYS_LAST_DUE_1ST_VERSION"].replace(365243, np.nan, inplace=True)
    prev["DAYS_LAST_DUE"].replace(365243, np.nan, inplace=True)
    prev["DAYS_TERMINATION"].replace(365243, np.nan, inplace=True)

    prev.columns = [
        "PREV_" + col if col not in ["SK_ID_PREV", "SK_ID_CURR"] else col
        for col in prev.columns
    ]
    prev, prev_cat = one_hot_encoder(prev, nan_as_category)

    return prev, prev.columns.to_list(), prev_cat


def previous_applications_engin_feats(df, prev_cols, prev_cat):
    prev = df[prev_cols]
    pos = remove_prefix(pos, "PREV_")
    # Add feature: value ask / value received percentage
    prev["APP_CREDIT_PERC"] = prev["AMT_APPLICATION"] / prev["AMT_CREDIT"]
    # Previous applications numeric features
    num_aggregations = {
        "AMT_ANNUITY": ["min", "max", "mean"],
        "AMT_APPLICATION": ["min", "max", "mean"],
        "AMT_CREDIT": ["min", "max", "mean"],
        "APP_CREDIT_PERC": ["min", "max", "mean", "var"],
        "AMT_DOWN_PAYMENT": ["min", "max", "mean"],
        "AMT_GOODS_PRICE": ["min", "max", "mean"],
        "HOUR_APPR_PROCESS_START": ["min", "max", "mean"],
        "RATE_DOWN_PAYMENT": ["min", "max", "mean"],
        "DAYS_DECISION": ["min", "max", "mean"],
        "CNT_PAYMENT": ["mean", "sum"],
    }
    # Previous applications categorical features
    cat_aggregations = {}

    for cat in prev_cat:
        cat_aggregations[cat] = ["mean"]

    prev_agg = prev.groupby("SK_ID_CURR").agg({**num_aggregations, **cat_aggregations})
    prev_agg.columns = pd.Index(
        ["PREV_" + e[0] + "_" + e[1].upper() for e in prev_agg.columns.tolist()]
    )
    # Previous Applications: Approved Applications - only numerical features
    approved = prev[prev["NAME_CONTRACT_STATUS_Approved"] == 1]
    approved_agg = approved.groupby("SK_ID_CURR").agg(num_aggregations)
    approved_agg.columns = pd.Index(
        ["APPROVED_" + e[0] + "_" + e[1].upper() for e in approved_agg.columns.tolist()]
    )
    prev_agg = prev_agg.join(approved_agg, how="left", on="SK_ID_CURR")
    # Previous Applications: Refused Applications - only numerical features
    refused = prev[prev["NAME_CONTRACT_STATUS_Refused"] == 1]
    refused_agg = refused.groupby("SK_ID_CURR").agg(num_aggregations)
    refused_agg.columns = pd.Index(
        ["REFUSED_" + e[0] + "_" + e[1].upper() for e in refused_agg.columns.tolist()]
    )
    prev_agg = prev_agg.join(refused_agg, how="left", on="SK_ID_CURR")

    del refused, refused_agg, approved, approved_agg, prev
    gc.collect()

    return prev_agg


# Preprocess
def preprocess(df, impute=False, smote=False):
    df = application(df)
    bureau, bureau_cols, bb_cols, bureau_cat, bb_cat = bureau_and_balance()
    df = df.join(bureau, how="left", on="SK_ID_CURR")

    del bureau
    gc.collect()

    prev, prev_cols, prev_cat = previous_applications()
    df = df.join(prev, how="left", on="SK_ID_CURR")

    del prev
    gc.collect()

    pos, pos_cols, pos_cat = pos_cash()
    df = df.join(pos, how="left", on="SK_ID_CURR")

    del pos
    gc.collect()

    ins, ins_cols, ins_cat = installments_payments()
    df = df.join(ins, how="left", on="SK_ID_CURR")

    del ins
    gc.collect()

    cc, cc_cols = credit_card_balance()
    df = df.join(cc, how="left", on="SK_ID_CURR")

    del cc
    gc.collect()

    df = df.reset_index(drop=True)

    # Iterate over the 'object' columns
    for col_name in df.select_dtypes(include="object").columns:
        # Replace True with 1.0 and False with 0.0
        df[col_name] = df[col_name].replace({True: 1.0, False: 0.0})
        # Change the column data type to float64
        df[col_name] = df[col_name].astype("float64")

    # Replace infinite values from division by zero with NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Find special JSON characters in column names
    pattern = re.compile(r"[:/,]|[\s]")
    # Replace colons, slashes, and commas with hyphens, and whitespace characters with underscores
    df.columns = [
        re.sub(pattern, lambda x: "-" if x.group(0) in ":/,," else "_", col)
        for col in df.columns.to_list()
    ]

    # Convert boolean values to integers
    bool_cols = [
        x
        for x in df.columns.to_list()
        if df[x].dtype == "bool" or sorted(df[x].unique().tolist()) == [0, 1]
    ]
    df[bool_cols] = df[bool_cols].astype(int)

    # Smote does not accept missing values; they must be imputed
    if smote or impute:
        # Create and fit the knn imputer
        imputed_data = KNNImputer(n_neighbors=5).fit_transform(df)

        # Convert the imputed data back to a DataFrame
        df = pd.DataFrame(imputed_data, columns=df.columns, index=df.index)

        # Round the imputed values for boolean columns
        df[bool_cols] = df[bool_cols].round().astype(int)

    # Apply SMOTE if the parameter is True
    if smote:
        smote = SMOTE(random_state=0)
        X_resampled, y_resampled = smote.fit_resample(
            df.drop(columns=["TARGET"]), df["TARGET"]
        )
        df = pd.concat([X_resampled, y_resampled], axis=1)

    # # Add a stratification column with the same values as the target variable
    # preproc_app_train_df["stratify_col"] = preproc_app_train_df["TARGET"]

    # # Perform a stratified train/test split using the stratify_col column
    # preproc_app_train_df, preproc_app_valid_df = train_test_split(
    #     preproc_app_train_df,
    #     test_size=0.2,
    #     random_state=0,
    #     stratify=preproc_app_train_df["stratify_col"],
    # )

    # # Drop the stratification column from both DataFrames
    # preproc_app_train_df = preproc_app_train_df.drop(
    #     columns=["stratify_col"]
    # ).reset_index(drop=True)
    # preproc_app_valid_df = preproc_app_valid_df.drop(
    #     columns=["stratify_col"]
    # ).reset_index(drop=True)
