# BLENDED_LERNING
# Implementation-of-Multiple-Linear-Regression-Model-with-Cross-Validation-for-Predicting-Car-Prices

## AIM:
To write a program to predict the price of cars using a multiple linear regression model and evaluate the model performance using cross-validation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import Libraries:
Bring in the necessary libraries.

2.Load the Dataset:
Load the dataset into your environment.

3.Data Preprocessing:
Handle any missing data and encode categorical variables as needed.

4.Define Features and Target:
Split the dataset into features (X) and the target variable (y).

5.Split Data:
Divide the dataset into training and testing sets.

6.Build Multiple Linear Regression Model:
Initialize and create a multiple linear regression model.

7.Train the Model:
Fit the model to the training data.

8.Evaluate Performance:
Assess the model's performance using cross-validation.

9.Display Model Parameters:
Output the model’s coefficients and intercept.

10.Make Predictions & Compare:
Predict outcomes and compare them to the actual values.


## Program:
```

Program to implement the multiple linear regression model for predicting car prices with cross-validation.
Developed by: S Harish
RegisterNumber:  212223040062





import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 1. Load and prepare data
data = pd.read_csv("CarPrice_Assignment (1).csv")

# Simple preprocessing - Remove unnecessary columns
data = data.drop(['car_ID', 'CarName'], axis=1)

# Handle categorical variables
data = pd.get_dummies(data, drop_first=True)

# 2. Split data
X = data.drop('price', axis=1)
y = data['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Create and train model
model = LinearRegression()
model.fit(X_train, y_train)

# 4. Evaluate with cross-validation (simple version)
from sklearn.model_selection import cross_val_score

print("=== Cross-Validation ===")
cv_scores = cross_val_score(model, X, y, cv=5)

print("Fold R² scores:", [f"{score:.4f}" for score in cv_scores])
print(f"Average R²: {cv_scores.mean():.4f}")

# 5. Test set evaluation
y_pred = model.predict(X_test)
print("\n=== Test Set Performance ===")
print(f"MSE: {mean_squared_error(y_test, y_pred):.2f}")
print(f"R²: {r2_score(y_test, y_pred):.4f}")

# 6. Visualization
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')  # Perfect prediction line
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Prices")
plt.grid(True)
plt.show()





```

## Output:

![image](https://github.com/user-attachments/assets/475f3a46-14b5-428d-b0fc-6478d6308919)

![simple linear regression model for predicting the marks scored](sam.png)


## Result:
Thus, the program to implement the multiple linear regression model with cross-validation for predicting car prices is written and verified using Python programming.
