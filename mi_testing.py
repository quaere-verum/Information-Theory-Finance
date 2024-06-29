# In this file, we perform tests on time series features like moving averages, RSI, MACD, etc.
# We test whether there is any significant mutual information between these indicators,
# and the target, i.e. some future outcome.
from information_theory import discrete_mutual_information, continuous_binned_mutual_information
import numpy as np
import pandas as pd
from typing import List, Callable
import os
DATAFOLDER = os.getenv('DATAFOLDER')


def fetch_data(ticker: str, start: str, end: str, interval: str) -> pd.DataFrame:
    # Modify as needed, using yfinance or your local data storage
    data = pd.read_csv(os.path.join(DATAFOLDER, f'{ticker.upper()}_full_{interval}.txt'),
                       usecols=[0, 4], 
                       names=['timestamp', 'close'], 
                       index_col=0, 
                       parse_dates=[0])
    return data.loc[start:end]


def test_feature(feature: Callable[[pd.Series], pd.Series], 
                 target: Callable[[pd.Series], pd.Series], 
                 feature_bin_quantiles: np.ndarray,
                 ticker: str,
                 start: str,
                 end: str,
                 interval: str):
    ticker = fetch_data(ticker=ticker, start=start, end=end, interval=interval)
    x = feature(ticker).dropna()
    y = target(ticker).dropna()
    intersection = x.index.intersection(y.index)
    x = x.loc[intersection]
    y = y.loc[intersection].astype(np.int32)
    return continuous_binned_mutual_information(x=x.values.reshape(-1), 
                                                y=y.values.reshape(-1), 
                                                x_quantiles=feature_bin_quantiles, 
                                                y_quantiles=None)


def make_rolling_mean(window: int) -> Callable[[pd.Series], pd.Series]:
    def rolling_mean(s: pd.Series) -> pd.Series:
        rm = s.rolling(window=window).mean()
        return s - rm
    return rolling_mean


def make_target_labels(take_profit: float, stop_loss: float, lookahead_window: int) -> Callable[[pd.Series], pd.Series]:
    def label(x: np.ndarray) -> float:
        future_returns = x[::-1] / x[-1] - 1
        time_to_profit = np.argmax(future_returns >= take_profit)
        time_to_loss = np.argmax(future_returns <= stop_loss)
        if time_to_loss == time_to_profit == 0:
            return 0
        elif time_to_profit == 0:
            return 0
        elif time_to_loss == 0:
            return 1
        elif time_to_profit < time_to_loss:
            return 1
        else:
            return 0
    def target_labels(s: pd.Series) -> pd.Series:
        return s.sort_index(ascending=False).rolling(window=lookahead_window).apply(label, raw=True).sort_index()
    return target_labels



if __name__ == '__main__':
    labeler = make_target_labels(0.01, -0.01, 6)
    rolling_mean = make_rolling_mean(12)
    print(test_feature(rolling_mean, labeler, 10, 'ETH', '2019-01-01', None, '1hour'))