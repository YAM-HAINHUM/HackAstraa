# STEP 1: IMPORT LIBRARIESmonthly_data = df.resample('ME').mean()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
# STEP 2: CREATE TIME SERIES DATAmonthly_data = df.resample('ME').mean()
# Generate sample time series data
date_range = pd.date_range(start='2023-01-01', periods=100, freq='D')
data = np.random.randn(100)
df = pd.DataFrame(data, index=date_range, columns=['Value'])

# Plot original data
plt.plot(df)
plt.title("Original Time Series Data")
plt.show()
monthly_data = df.resample('ME').mean()
# STEP 3: CHECK STATIONARITY (ADF TEST)monthly_data = df.resample('ME').mean()
result = adfuller(df['Value'])

print("ADF Statistic:", result[0])
print("p-value:", result[1])
monthly_data = df.resample('ME').mean()
# STEP 4: DIFFERENCING (IF NOT STATIONARY)monthly_data = df.resample('ME').mean()
df_diff = df.diff().dropna()

plt.plot(df_diff)
plt.title("Differenced Data")
plt.show()
monthly_data = df.resample('ME').mean()
# STEP 5: ACF & PACF PLOTSmonthly_data = df.resample('ME').mean()
plot_acf(df_diff)
plt.show()

plot_pacf(df_diff)
plt.show()
monthly_data = df.resample('ME').mean()
# STEP 6: BUILD ARIMA MODEL
# (p=1, d=1, q=1)monthly_data = df.resample('ME').mean()
model = ARIMA(df['Value'], order=(1, 1, 1))
model_fit = model.fit()

print(model_fit.summary())
monthly_data = df.resample('ME').mean()
# STEP 7: FORECASTmonthly_data = df.resample('ME').mean()
forecast = model_fit.forecast(steps=10)

print("Forecasted Values:")
print(forecast)
monthly_data = df.resample('ME').mean()
# STEP 8: PLOT FORECASTmonthly_data = df.resample('ME').mean()
plt.plot(df, label='Original Data')
plt.plot(forecast, label='Forecast', color='red')

plt.title("ARIMA Forecast")
plt.legend()
plt.show()