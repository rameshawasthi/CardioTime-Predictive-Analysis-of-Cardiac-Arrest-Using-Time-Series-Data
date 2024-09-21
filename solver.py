# %%
import datetime
import os
from pathlib import Path
from statistics import LinearRegression

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from imblearn.over_sampling import SMOTE
from matplotlib import pyplot
from sklearn import metrics
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras import layers, losses, metrics, optimizers
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout
from typing_extensions import dataclass_transform

from utils import plot_history, plot_auc, get_confusion_metrics
from constants import *

# from pandas import Dataframe

# %%

keras.utils.set_random_seed(12345)


# %%
HOURLY_DATA_PATH = "timeseries.csv"
METADATA_PATH = "meta_data.csv"

# %%
df_hr = pd.read_csv(HOURLY_DATA_PATH, index_col=0, parse_dates=["event_time"])
df_hr.head()
# %%
meta_df = pd.read_csv(METADATA_PATH, index_col=0, parse_dates=["event_time"])
meta_df.head()
# %%
JOIN_KEY = ["subject_id", "hadm_id", "icustay_id"]
req_cols = set(meta_df.columns) - set(df_hr.columns)
req_cols = list(req_cols) + JOIN_KEY
join_df = df_hr.merge(
    meta_df[req_cols],
    on=JOIN_KEY,
    how="left",
)
join_df.head()

#%%

# MICE method
def mice_impute_data(train):
    lr_for_impute = LinearRegression()
    imputer = IterativeImputer(
        estimator=lr_for_impute,
        tol=1e-10,
        max_iter=30,
        verbose=0,
        imputation_order="roman",
    )
    X_train = imputer.fit_transform(train)
    # X_test = imputer.transform(test)
    return pd.DataFrame(X_train, columns=train.columns)


# %%
join_df = join_df.drop(columns=["event_time"])
join_df_imputed = mice_impute_data(join_df.copy())

# %%
dataset = []
for i, (group, group_df) in enumerate(join_df_imputed.groupby(JOIN_KEY)):
    X_ts = group_df[list(X_COLS_TS)].copy()
    X_base = group_df[list(X_COLS_BASE)].copy().drop_duplicates()
    if len(X_base) > 1:
        X_base = X_base.iloc[0]
    Y_group = group_df["class"].iloc[0]
    dataset.append(tuple((i, group, X_ts, X_base, Y_group)))


# %%
# X_data = join_df[X_COLS].copy()
# Y_data = join_df[["class"]].copy()
# %%
data_train, data_test = train_test_split(dataset, train_size=0.8, random_state=24)


# %%
import tqdm


def smote(train_data):

    all_dfs = []
    all_y = []
    for i, ids, X_ts, X_base, y in tqdm.tqdm(train_data):
        df_ts = X_ts.copy()
        df_ts["index"] = i
        df_ts["index"] = df_ts["index"].astype(int)
        df_base = X_base.copy()
        df_base = pd.DataFrame(df_base)
        df_base["index"] = i
        df_base["index"] = df_base["index"].astype(int)
        df = df_ts.merge(df_base, how="left", on="index")
        if df.shape != (30, 35):
            continue
        all_dfs.append(df)
        all_y.extend([y for _ in range(30)])

    df_x = pd.concat(all_dfs, axis=0)

    # df_x = mice_impute_data(df_x)
    print(df_x.shape, len(all_y))
    smoter = SMOTE(random_state=2)
    df_x_smote, y_smote = smoter.fit_resample(df_x, all_y)
    print(df_x_smote.shape, len(y_smote))

    smoted_data = []
    for i, ids, X_ts, X_base, y in tqdm.tqdm(train_data):
        req_data = df_x_smote[df_x_smote["index"] == i].copy()
        if (req_data.shape) != (30, 35):
            continue
        # print(req_data.columns)
        X_ts = req_data[list(X_COLS_TS)]
        X_base = req_data[list(X_COLS_BASE)]
        smoted_data.append(tuple((i, ids, X_ts, X_base, Y_group)))

    return smoted_data


smoted_dataset = smote(data_train)

# %%
data_train_filtered = filter(lambda x: x[2].shape == (30, 8) and len(x[3]), data_train)
data_train_filtered = [x for x in tqdm.tqdm(data_train_filtered)]

data_test_filtered = filter(lambda x: x[2].shape == (30, 8) and len(x[3]), data_test)
data_test_filtered = [x for x in tqdm.tqdm(data_test_filtered)]


# %%
def get_model(use_bidirectional=False):
    input_ts = keras.Input(shape=(30, 8), name="timeseries")
    input_mlp = keras.Input(shape=(26,), name="baseline")
    dense1_mlp = Dense(10, activation="relu")(input_mlp)
    dropout1_mlp = Dropout(0.8)(dense1_mlp)
    output_mlp = Dense(2)(dropout1_mlp)
    rnn_layer = LSTM(20, dropout=0.6, recurrent_dropout=0.8, activation="tanh")
    if use_bidirectional:
        rnn_ts = Bidirectional(rnn_layer)(input_ts)
    else:
        rnn_ts = rnn_layer(input_ts)
    output_ts = Dense(1)(rnn_ts)

    x = layers.Concatenate()([output_ts, output_mlp])
    x = Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs=[input_ts, input_mlp], outputs=[x])

    print(model.summary())

    return model


# %%
def train(model, data_train, data_test):
    model.compile(
        loss=losses.BinaryCrossentropy(from_logits=True),
        optimizer=optimizers.Adam(
            learning_rate=1e-3,
            epsilon=1e-7,
        ),
        metrics=[metrics.Recall(), metrics.Precision(), metrics.AUC()],
    )
    history = model.fit(
        {
            "timeseries": np.array([x[2].values for x in data_train]),
            "baseline": np.array([x[3].values.flatten() for x in data_train]),
        },
        np.array([x[-1] for x in data_train]),
        validation_data=(
            {
                "timeseries": np.array([x[2].values for x in data_test]),
                "baseline": np.array([x[3].values.flatten() for x in data_test]),
            },
            np.array([x[-1] for x in data_test]),
        ),
        # batch_size=128,
        epochs=50,
    )
    return model, history


# %%

model_sota = get_model(use_bidirectional=False)


model_sota, history_sota = train(model_sota, data_train_filtered, data_test_filtered)

to_plot = {x.replace("val_", "") for x in history_sota.history.keys()}
model_sota.save("sota_model.h5")
for metric in to_plot:
    fig, ax = plt.subplots()
    plot_history(fig, ax, history_sota, metric, suffix="LSTM")
    plt.show()


y_test = np.array([x[-1] for x in data_test_filtered])
y_pred = model_sota.predict(
    {
        "timeseries": np.array([x[2].values for x in data_test_filtered]),
        "baseline": np.array([x[3].values.flatten() for x in data_test_filtered]),
    },
).ravel()


fig, ax = plt.subplots()
plot_auc(fig, ax, y_test, y_pred, suffix="LSTM")
plt.show()

# %%

THRESHOLD = np.percentile(y_pred, 80)
y_pred_flat = np.where(y_pred >= THRESHOLD, 1, 0)
sota_metrics = get_confusion_metrics(y_test, y_pred_flat)

# %%

model_proposed = get_model(use_bidirectional=True)


model_proposed, history_proposed = train(
    model_proposed, data_train_filtered, data_test_filtered
)

to_plot = {x.replace("val_", "") for x in history_proposed.history.keys()}
model_proposed.save("model_proposed.h5")

for metric in to_plot:
    fig, ax = plt.subplots()
    plot_history(fig, ax, history_proposed, metric, suffix="BiDirectional")
    plt.show()


y_test = np.array([x[-1] for x in data_test_filtered])
y_pred_p = model_proposed.predict(
    {
        "timeseries": np.array([x[2].values for x in data_test_filtered]),
        "baseline": np.array([x[3].values.flatten() for x in data_test_filtered]),
    },
).ravel()

fig, ax = plt.subplots()
plot_auc(fig, ax, y_test, y_pred_p, suffix="BiDirectional")
plt.show()

# %%
THRESHOLD = np.percentile(y_pred_p, 80)
y_pred_flat_p = np.where(y_pred_p >= THRESHOLD, 1, 0)

bilstm_metrics = get_confusion_metrics(y_test, y_pred_flat_p)

# %%
print("checkpoint")

labels = ["accuracy", "sensitivity", "specificity"]

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width / 2, sota_metrics.values(), width, label="State of Art")
rects2 = ax.bar(x + width / 2, bilstm_metrics.values(), width, label="Proposed Model")


# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel("")
ax.set_title(
    "Comparing the results of CA Prediction between state of art and proposed solution"
)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
plt.show()

# %%

# %%

# %%
