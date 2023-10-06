# importing required libraries
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split 
import pandas as pd

airbnb_users_train_df = pd.read_csv("processed_df.csv")

# setting up x and y values
X = airbnb_users_train_df.drop(columns='country_destination',axis=1)
y = airbnb_users_train_df['country_destination']

# Splitting the X and y into X_train, y_train, X_test and y_test using the train_test_split function.
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=1)

# Creating an object xgbModel of XBGClassifier
xgbModel = XGBClassifier()

# Using the fit function from XBGClassifier function, data has been fitted with default parameters.
xgbModel.fit(X_train,y_train)

# # Making predictions on the User Input Data
# xgbPredictions = xgbModel.predict(X_test)



