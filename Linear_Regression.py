import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

# Importing the CSV file
dataset = pd.read_csv('sat_gpa.csv')

x = dataset.iloc[:, 0:2].values 
y = dataset.iloc[:, 1].values 

from sklearn.model_selection import train_test_split 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30,\
random_state=0)

from sklearn.linear_model import LinearRegression 
regression = LinearRegression()

# Fitting the model on the training data
regression.fit(x_train, y_train) 

# Getting the intercept, coefficient and predicted values
intercept = regression.intercept_
coef = regression.coef_
y_predict = regression.predict(x_test)

# Getting the R-squared score on the training and testing data
train_score = regression.score(x_train, y_train)
test_score = regression.score(x_test, y_test)

print("Intercept:", intercept)
print("Coefficient:", coef)
print("Predicted values:", y_predict)
print("Training score:", train_score)
print("Testing score:", test_score)
