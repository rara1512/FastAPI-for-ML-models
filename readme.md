#This is a tutorial on how to use FastAPI to deploy ML models.#
In our case, we have used XGBRegressor for making house price predictions and the data is given as df_train and df_test.
XGB_model.py contains the code to train the model and store it in a pickle file using joblib python package.
FastAPI_Practice.py contains the code to setup the FastAPI functionality and the code to start a local server that can be used for API calls.
This has to be run using command-
uvicorn FastAPI_Practice:app --reload
Once this is running, we need to run test_api.py to call the API using one of the records as input to get model predictions
