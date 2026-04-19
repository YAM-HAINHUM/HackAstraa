# STEP 1: IMPORT LIBRARIES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# STEP 2: CREATE / LOAD TIME SERIES DATA
# Create sample time series data
date_range = pd.date_range(start='2023-01-01', periods=100, freq='D')
data = np.random.randn(100)
df = pd.DataFrame(data, index=date_range, columns=['Value'])
print(df.head())

# STEP 3: INDEXING (DatetimeIndex)
# Already indexed by date
df.index = pd.to_datetime(df.index)

# STEP 4: RESAMPLING
# Convert daily data to monthly average
monthly_data = df.resample('ME').mean()
print(monthly_data)

# STEP 5: ROLLING ANALYSIS
# Rolling mean (window = 7 days)
df['Rolling_Mean'] = df['Value'].rolling(window=7).mean()

# Rolling standard deviation
df['Rolling_Std'] = df['Value'].rolling(window=7).std()

# STEP 6: PLOTTING TIME SERIES
plt.figure(figsize=(10,5))

# Original data
plt.plot(df['Value'], label='Original Data')
# Rolling Mean
plt.plot(df['Rolling_Mean'], label='Rolling Mean (7 days)', color='red')
# Rolling Std
plt.plot(df['Rolling_Std'], label='Rolling Std (7 days)', color='green')
plt.title("Time Series Analysis")
plt.xlabel("Date")
plt.ylabel("Values")
plt.legend()
plt.show()

# STEP 7: PLOT RESAMPLED DATA
plt.figure(figsize=(10,5))
plt.plot(monthly_data, marker='o')
plt.title("Monthly Resampled Data")
plt.xlabel("Month")
plt.ylabel("Average Value")
plt.show()