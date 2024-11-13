# Import Packages
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import root_mean_squared_error
import joblib

# Load and preprocess the data
df_train = pd.read_csv('Regression_data/df_train.csv')
df_test = pd.read_csv('Regression_data/df_test.csv')

target_column = 'price'

X_train = df_train.drop(target_column, axis=1)
y_train = df_train[target_column]
X_test = df_test.drop(target_column, axis=1)
y_test = df_test[target_column]

# Model Training
model = XGBRegressor(random_state = 42)
model.fit(X_train, y_train)

# Evaluate model
predictions = model.predict(X_test)
rmse = root_mean_squared_error(y_test, predictions)
print(f"RMSE: {rmse}")

# Dump the model in a pickle file
joblib.dump(model, "xgboost_regressor_model.pkl")

# Print completion message
print('File execution complete')