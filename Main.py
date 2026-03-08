#import libraries
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression 
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

#download raw data
def download_data(ticker="NVDA", start="2010-01-01"):
    df = yf.download(ticker, start)
    df = df.sort_index()
    return df
df = download_data()
print(df.head())

#calculate returns
df['Return'] = df['Adj Close'].pct_change()
df = df.dropna()

#create target
df['Target'] = (df['Return'].shift(-1) > 0).astype(int)
df = df.dropna()

df["MA_5"] = df["Adj Close"].rolling(5).mean()
df["MA_20"] = df["Adj Close"].rolling(20).mean()

#adding momentum and volatility
df["Momentum_5"] = df["Adj Close"] - df["Adj Close"].shift(5)
df["Volatility_5"] = df["Return"].rolling(5).std()
df["Volume_Change"] = df["Volume"].pct_change()

#train/test split
split_in = int(len(df)*0.8)
train = df.iloc[split_in]
test = df.iloc[split_in:]

#baseline logistic regression
features = ["MA_5", "MA_20", "Momentum_5", "Volatility_5", "Volume_Change"]
X_train = train[features]
y_train = train["Target"]

X_train_scaled = StandardScaler.fit_transform(X_train)
model = LogisticRegression(max_iter=2000)
model.fit(X_train_scaled, y_train)

#evaluate logistic regression
X_test = test[features]
y_test = test["Target"]
X_test_scaled = StandardScaler.fit_transform(X_test)

predictions = model.predict(X_test_scaled)

print("Accuracy:", accuracy_score(y_test, predictions))
print("Precision:", precision_score(y_test, predictions))
print("Recall:", recall_score(y_test, predictions))
print("F1 Score:", f1_score(y_test, predictions))
print("Confusion MAtrix:\n", confusion_matrix(y_test, predictions))

#backtesting logic
test = test.copy()
test["Position"] = predictions
test["Position"] = test["Position"].shift(1)
test["Strategy_Return"] = test["Position"] * test["Return"]
test["Cumulative_Strategy"] = (1 + test["Strategy_Return"]).cumprod()
test["Cumulative_BuyHold"] = (1 + test["Return"]).cumprod()

#Computing Sharpe & Max Drawdown
returns = test["Strategy_Retun"].dropna()
sharpe = (returns.mean() / returns.std()) * np.sqrt(252)
cummulative = test["Cummulative_Strategy"]
peak = cummulative.cummax()
drawdown = (cummulative - peak) / peak
max_drawdown = drawdown.min()
print("Sharpe Ratio:", sharpe)
print("Max Drawdown:", max_drawdown)

#plot equity curve
plt.figure(figsize=(12, 8))
plt.plot(test["Cummulative_Strategy"], label="Strategy")
plt.plot(test["Cummulative_BuyHold"], label="Buy & Hold")
plt.title("Equity_curve")
plt.legend()
plt.show()

