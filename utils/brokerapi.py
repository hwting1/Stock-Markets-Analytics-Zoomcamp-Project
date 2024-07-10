import numpy as np
import yfinance as yf

class StockBrokerAPI:

    def __init__(self, tickers):
        self._tickers = tickers
        self._hold_shares = np.zeros(len(tickers))
        self._order_history = []
        self._trading_date = []

    def get_data(self, start_date, end_date):
        data = {}
        for ticker in self._tickers:
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)
            df.reset_index(inplace=True)
            data[ticker] = df
        return data

    def execute_order(self, weight_change, date):
        self._hold_shares += weight_change
        self._order_history.append(weight_change)
        self._trading_date.append(date)

    def reset(self, tickers):
        self._tickers = tickers
        self._hold_shares = np.zeros(len(tickers))
        self._order_history = []
        self._trading_date = []

    @property
    def hold_shares(self):
        return self._hold_shares

    @property
    def order_history(self):
        return self._order_history

    @property
    def trading_date(self):
        return self._trading_date

    @property
    def tickers(self):
        return self._tickers
