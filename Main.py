#import libraries
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
df("MA_20") = df["Adj Close"].rolling(20).mean()
