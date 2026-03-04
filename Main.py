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


