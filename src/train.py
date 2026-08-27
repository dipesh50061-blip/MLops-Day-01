import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib


df = pd.read_csv("C:\\Users\\awara\\Downloads\\MLops Day-01\\data\\Advertising.csv",index_col=0)

X,y = df.drop(columns=['sales']), df['sales']
xtrain, xtest, ytrain, ytest = train_test_split(X,y,test_size=0.2, random_state=67)

model = LinearRegression()
model.fit(xtrain, ytrain)

joblib.dump(model, "models\\linear_regression_model.pkl")