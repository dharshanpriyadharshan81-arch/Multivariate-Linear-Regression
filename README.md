# Implementation of Multivariate Linear Regression
## Aim
To write a python program to implement multivariate linear regression and predict the output.
## Equipment’s required:
1.	Hardware – PCs
2.	Anaconda – Python 3.7 Installation / Moodle-Code Runner
## Algorithm:
### Step1
<br>

### Step2
<br>

### Step3
<br>

### Step4
<br>

### Step5
<br>

## Program:
```
# Import required libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

# Load dataset
housing = fetch_california_housing()
X = housing.data
y = housing.target

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=1
)

# Create and train Linear Regression model
reg = LinearRegression()
reg.fit(X_train, y_train)

# Print coefficients and variance score
print("Coefficients:", reg.coef_)
print("Variance score (R^2):", reg.score(X_test, y_test))
# Plot residual errors
plt.style.use('fivethirtyeight')

# Training data residuals
plt.scatter(
    reg.predict(X_train),
    reg.predict(X_train) - y_train,
    color="green",
    s=10,
    label="Train data"
)
# Testing data residuals
plt.scatter(
    reg.predict(X_test),
    reg.predict(X_test) - y_test,
    color="blue",
    s=10,
    label="Test data"
)

# Zero residual line
plt.hlines(y=0, xmin=0, xmax=5, linewidth=2)

# Plot settings
plt.legend(loc="upper right")
plt.title("Residual errors")
plt.show()


```
## Output:
<img width="1403" height="761" alt="Screenshot 2025-12-27 133243" src="https://github.com/user-attachments/assets/897f9715-999e-4f7e-86e5-7cc5a8aba57c" />

### Insert your output

<br>

## Result
Thus the multivariate linear regression is implemented and predicted the output using python program.
