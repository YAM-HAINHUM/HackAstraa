# STEP 1: IMPORT LIBRARIES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

import kagglehub

# STEP 2: LOAD DATASET FROM KAGGLE
path = kagglehub.dataset_download("mirichoi0218/insurance")

print("Dataset Path:", path)

# Read CSV file
df = pd.read_csv(path + "/insurance.csv")

# Display data
print(df.head())

# STEP 3: DATA ANALYSIS
print(df.info())
print(df.describe())

# Visualize relationship (age vs charges)
sns.scatterplot(x='age', y='charges', data=df)
plt.title("Age vs Charges")
plt.show()

# STEP 4: SELECT VARIABLES
# (Simple Linear Regression: only ONE input)
X = df[['age']]        # Independent variable
y = df['charges']      # Dependent variable

# STEP 5: TRAIN-TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

# STEP 6: TRAIN MODEL
model = LinearRegression()
model.fit(X_train, y_train)

# STEP 7: PREDICTION
y_pred = model.predict(X_test)

# STEP 8: VISUALIZATION

# Training Set
plt.scatter(X_train, y_train, color='blue')
plt.plot(X_train, model.predict(X_train), color='red')
plt.title("Training Set (Age vs Charges)")
plt.xlabel("Age")
plt.ylabel("Charges")
plt.show()

# Test Set
plt.scatter(X_test, y_test, color='green')
plt.plot(X_train, model.predict(X_train), color='red')
plt.title("Test Set (Age vs Charges)")
plt.xlabel("Age")
plt.ylabel("Charges")
plt.show()

# MODEL PARAMETERS
print("Slope (m):", model.coef_[0])
print("Intercept (c):", model.intercept_)