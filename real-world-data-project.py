import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ==========================================
# 1. Data Collection (Finance - Stock Prices)
# ==========================================
print("Fetching real-world stock data (Apple Inc. - AAPL)...")
# Fetch 2 years of historical daily data
ticker = 'AAPL'
data = yf.download(ticker, period='2y', interval='1d')

# Drop missing values if any exist
data.dropna(inplace=True)

# ==========================================
# 2. Exploratory Data Analysis (EDA)
# ==========================================
# Calculate 50-day and 200-day Moving Averages
data['MA_50'] = data['Close'].rolling(window=50).mean()
data['MA_200'] = data['Close'].rolling(window=200).mean()

plt.figure(figsize=(14, 6))
plt.plot(data.index, data['Close'], label='Close Price', color='blue', alpha=0.6)
plt.plot(data.index, data['MA_50'], label='50-Day Moving Average', color='red', linestyle='--')
plt.plot(data.index, data['MA_200'], label='200-Day Moving Average', color='green', linestyle='--')
plt.title(f'{ticker} Stock Price and Moving Averages')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True)
plt.show()

# ==========================================
# 3. Feature Engineering & Prediction Setup
# ==========================================
# We will predict the next day's closing price based on today's Open, High, Low, Close, and Volume.
print("\nPreparing data for predictive modeling...")
data['Target'] = data['Close'].shift(-1) # Next day's close price is the target
data.dropna(inplace=True) # Drop the last row since it won't have a target

# Define features (X) and target (y)
features = ['Open', 'High', 'Low', 'Close', 'Volume']
X = data[features]
y = data['Target']

# Split into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False) # Time series requires shuffle=False

# ==========================================
# 4. Model Training & Evaluation
# ==========================================
print("Training Linear Regression Model...")
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"\n--- Model Evaluation Metrics ---")
print(f"Mean Absolute Error (MAE): ${mae:.2f}")
print(f"Root Mean Squared Error (RMSE): ${rmse:.2f}")
print(f"R-squared Score: {r2:.4f}")

# ==========================================
# 5. Visualizing Predictions vs Reality
# ==========================================
plt.figure(figsize=(14, 6))
# Only plotting the test data timeline
plt.plot(y_test.index, y_test.values, label='Actual Closing Price', color='blue')
plt.plot(y_test.index, y_pred, label='Predicted Closing Price', color='orange', linestyle='dashed')
plt.title(f'{ticker} Actual vs Predicted Stock Prices')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True)
plt.show()

# ==========================================
# 6. Structured Conclusions
# ==========================================
print("\n" + "="*50)
print(" FINAL REPORT & CONCLUSIONS ")
print("="*50)
print("1. Data Trends: The EDA plots revealed distinct upward/downward trends. Moving averages helped smooth out daily volatility to identify broader market direction.")
print(f"2. Predictive Accuracy: The Linear Regression model achieved an R-squared of {r2:.2f}, indicating how well the current day's features explain the next day's variance.")
print(f"3. Error Margin: On average, the model's predictions are off by roughly ${mae:.2f} per share (MAE).")
print("4. Limitations: While the model tracks general movements well, linear regression struggles to predict sudden market shocks or non-linear stock market behaviors.")
print("="*50 + "\n")