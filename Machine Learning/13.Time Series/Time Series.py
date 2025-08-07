# Step 0: Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose

# Step 1: Load the dataset from an online source or manually create it
df = pd.read_csv('airline-passengers.csv')

# Step 2: Parse dates and set as index
df['Month'] = pd.to_datetime(df['Month'])
df.set_index('Month', inplace=True)

# Step 3: Basic line plot
plt.figure(figsize=(12, 6))
plt.plot(df['Passengers'], label='Monthly Passengers', linewidth=2)
plt.title("Airline Passenger Data (1949 - 1960)")
plt.xlabel("Year")
plt.ylabel("Number of Passengers")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Step 4: Add rolling average (12-month moving average)
df['Rolling_Mean'] = df['Passengers'].rolling(window=12).mean()

# Step 5: Plot original + rolling average
plt.figure(figsize=(12, 6))
plt.plot(df['Passengers'], label='Original', linewidth=2)
plt.plot(df['Rolling_Mean'], label='12-Month Rolling Average', color='red', linewidth=2)
plt.title("Airline Passengers with Rolling Average")
plt.xlabel("Year")
plt.ylabel("Passengers")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Step 6: Seasonal decomposition
decompose_result = seasonal_decompose(df['Passengers'], model='multiplicative')
decompose_result.plot()
plt.tight_layout()
plt.show()
