# STEP 1: IMPORT LIBRARIES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

import kagglehub

# STEP 2: LOAD DATASET
path = kagglehub.dataset_download("mirichoi0218/insurance")
df = pd.read_csv(path + "/insurance.csv")

print(df.head())

# STEP 3: HANDLE CATEGORICAL VARIABLES

# Convert categorical variables using One-Hot Encoding
df = pd.get_dummies(df, drop_first=True)

print(df.head())

# STEP 4: SPLIT VARIABLES

X = df.drop("charges", axis=1)   # Independent variables
y = df["charges"]                # Dependent variable

# STEP 5: TRAIN-TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

# STEP 6: TRAIN MODEL

model = LinearRegression()
model.fit(X_train, y_train)

# STEP 7: PREDICTION

y_pred = model.predict(X_test)

# MODEL EVALUATION

from sklearn.metrics import mean_squared_error, r2_score

print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# COEFFICIENTS

coeff_df = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])
print(coeff_df)

print("Intercept:", model.intercept_)

# VISUALIZATION (Actual vs Predicted)

plt.scatter(y_test, y_pred)
plt.xlabel("Actual Charges")
plt.ylabel("Predicted Charges")
plt.title("Actual vs Predicted (MLR)")
plt.show()