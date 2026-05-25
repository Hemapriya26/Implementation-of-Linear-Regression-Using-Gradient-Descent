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
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor

data = {
    'Size': [1000, 1200, 1500, 1800, 2000],
    'Price': [300000, 350000, 400000, 450000, 500000]
}

df = pd.DataFrame(data)

X = df[['Size']]
y = df['Price']

model = SGDRegressor()

model.fit(X, y)

prediction = model.predict([[1600]])

print("Predicted Price:", prediction[0])

plt.scatter(X,y)

plt.plot(X, model.predict(X))

plt.xlabel("House Size")
plt.ylabel("House Price")
plt.title("House Price Prediction using SGD Regressor")

plt.show()
```

## Output:

<img width="1152" height="618" alt="image" src="https://github.com/user-attachments/assets/8c845906-86db-45f1-9253-2112e5398c03" />

## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
