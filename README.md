# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load the dataset and extract R&D Spend (X) and Profit (y) as input and output variables.

2.Normalize the input feature X using mean and standard deviation to improve convergence.

3.Initialize model parameters slope (m) and intercept (b) to zero.

4.For a fixed number of epochs, compute predicted values and calculate gradients of loss w.r.t m and b.

5.Update m and b using gradient descent with a chosen learning rate to minimize error.

6.Use the trained model to predict profit for new input and visualize results with a regression line.


## Program:
```
Program to implement the linear regression using gradient descent.
Developed by: Hemapriya P
RegisterNumber: 212225040126 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data = pd.read_csv("50_startups.csv")
print(data.head())
X = data['R&D Spend'].values
y = data['Profit'].values
X = (X - X.mean()) / X.std()
y = (y - y.mean()) / y.std()
m = 0
c = 0
L = 0.01
epochs = 1000
n = len(X)
for i in range(epochs):
    y_pred = m * X + c
    D_m = (-2/n) * sum(X * (y - y_pred))
    D_c = (-2/n) * sum(y - y_pred)
    m = m - L * D_m
    c = c - L * D_c
print("Slope (m):", m)
print("Intercept (c):", c)
y_pred = m * X + c
plt.scatter(X, y, color='blue')
plt.plot(X, y_pred, color='red')
plt.xlabel("R&D Spend")
plt.ylabel("Profit")
plt.title("Linear Regression using Gradient Descent")
plt.show()
```

## Output:

<img width="856" height="833" alt="image" src="https://github.com/user-attachments/assets/69f338d6-7995-4d52-bd05-7c5190ac2da1" />


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
