import pandas as pd
import requests

# Load the df_test.csv file
df_test = pd.read_csv("Regression_data/df_test.csv")

# Define the target column name
target_column = 'price'  # Replace with your actual target column name
# Modify the row to get its prediction
row_num = 5

# Select a single row (for example, the first row)
row = df_test.iloc[row_num]  # Change 0 to the index of the row you want to test

# Exclude the target column from the features
features_dict = row.drop(target_column).to_dict()

# Prepare the data for the API request
payload = {
    "features": features_dict
}

# Define the API endpoint
url = "http://127.0.0.1:8000/predict"

# Send the POST request to the FastAPI server
response = requests.post(url, json=payload)

# Check the response and print the prediction
if response.status_code == 200:
    prediction = response.json()
    print(f"Prediction for row {row_num}: {prediction['prediction']}")
else:
    print(f"Error: {response.status_code} - {response.text}")