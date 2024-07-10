import numpy as np

class DataTransformer:

    def __init__(self, tickers, seq_len):
        self.tickers = tickers
        self.seq_len = seq_len
        self.features = ['Norm_close', 'Norm_open', 'Norm_high', 'Norm_low', 'TRR']

    def transform(self, df_dict, prefix, save=True, interval=10):

        states_array = []
        trading_prices = []
        for ticker in self.tickers:
            df = df_dict[ticker]
            trading_price = df["Close"].values
            df["pre_Close"] = df["Close"].shift(1)
            df["TRR"] = (df["High"] - df["Low"]) / df["Close"] # True Range Ratio
            df["Norm_close"] = np.log( df["Close"] / df["pre_Close"])
            df["Norm_open"] = np.log(df["Open"] / df["pre_Close"])
            df["Norm_high"] = np.log(df["High"] / df["pre_Close"])
            df["Norm_low"] = np.log(df["Low"] / df["pre_Close"])
            df.dropna(inplace=True, axis=0)

            if prefix == "valid":
                state = df.iloc[:-1][self.features].values
                states_array.append(state)
            else:
                state = df[self.features].values
                states_array.append(state)

            trading_prices.append(trading_price)

        states_array = np.array(states_array)
        states_array = np.transpose(states_array, (1, 0, 2))
        trading_prices = np.array(trading_prices).T

        if prefix == "train":
            states_array, trading_prices = self._generate_dataset(states_array, trading_prices, interval)
        elif prefix == "valid":
            states_array = np.expand_dims(states_array, axis=0)
            trading_prices = np.expand_dims(trading_prices, axis=0)

        if save:
            np.save(f"data/{prefix}_data", states_array)
            np.save(f"data/{prefix}_trading_prices", trading_prices)
        else:
            return states_array, trading_prices

    def _generate_dataset(self, state_array, trading_prices, interval):
        days = state_array.shape[0]
        seq_num = days - self.seq_len + 1
        start_idx = np.arange(0, seq_num, interval)
        batch_states = []
        batch_prices = []
        for idx in start_idx:
            state = state_array[idx:idx + self.seq_len, :, :]
            price = trading_prices[idx:idx + self.seq_len, :]
            batch_states.append(state)
            batch_prices.append(price)

        return np.array(batch_states), np.array(batch_prices)


class StreamingDataTransformer:

    def __init__(self, tickers):
        self.tickers = tickers
        self.pre_Close = {ticker: None for ticker in tickers}
        self.features = ['Norm_close', 'Norm_open', 'Norm_high', 'Norm_low', 'TRR']

    def set_pre_close(self, df_dict):
        for ticker in self.tickers:
            df = df_dict[ticker]
            self.pre_Close[ticker] = df["Close"].values[-1]

    def transform(self, df_dict):
        state_arrays = []
        trading_prices = []
        for ticker in self.tickers:
            df = df_dict[ticker]
            pre_Close = self.pre_Close[ticker]
            assert df.shape[0] == 1
            df["TRR"] = (df["High"] - df["Low"]) / df["Close"] # True Range Ratio
            df["Norm_close"] = np.log( df["Close"] / pre_Close)
            df["Norm_open"] = np.log(df["Open"] / pre_Close)
            df["Norm_high"] = np.log(df["High"] / pre_Close)
            df["Norm_low"] = np.log(df["Low"] / pre_Close)

            state = df[self.features].values
            trading_price = df["Close"].values
            state_arrays.append(state)
            trading_prices.append(trading_price)
            self.pre_Close[ticker] = df["Close"].values

        states_array = np.array(state_arrays)
        trading_prices = np.array(trading_prices).reshape(1, -1)
        return states_array, trading_prices

