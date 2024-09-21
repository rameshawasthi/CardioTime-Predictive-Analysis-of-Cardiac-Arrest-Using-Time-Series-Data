import numpy as np
import pandas as pd
from tensorflow import keras

from constants import THRESHOLD
from constants import X_COLS_TS, X_COLS_BASE




timeseries = pd.read_csv("test_timeseries.csv")
timeseries = timeseries[X_COLS_TS]

baseline = pd.read_csv("test_baseline.csv")
baseline = baseline[X_COLS_BASE]

choice = input("Enter modelname (sota or bi-lstm): ")

if choice == "sota":
    model = keras.models.load_model(filepath="./sota_model.h5")
elif choice == "bi-lstm":
    model = keras.models.load_model(filepath="./model_proposed.h5")
else:
    raise ValueError("Modelname not valid")

prediction = model.predict({
    "timeseries": timeseries.values.reshape(1,-1,8),
    "baseline": baseline.values,
}
)

prediction = np.where(prediction > THRESHOLD, 1, 0)

prediction_str = "Heart Attack" if prediction else "Not Heart Attack"

print(prediction_str)
