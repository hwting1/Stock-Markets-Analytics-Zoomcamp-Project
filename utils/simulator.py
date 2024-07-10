import numpy as np
import pandas as pd
from tabulate import tabulate
class PortfolioSimulator:

    def __init__(self, init_cash):
        self.asset_value_history = []
        self.trading_date = []
        self.hold_shares = []
        self._now_cash = init_cash
        self.cash = []

    def record(self, hold_shares, trading_price, now_cash, date):
        self.trading_date.append(date)
        self.hold_shares.append(hold_shares)
        self.cash.append(now_cash)
        asset_value = np.sum(hold_shares * trading_price) + now_cash
        self._now_cash = now_cash
        self.asset_value_history.append(asset_value)

    @property
    def now_cash(self):
        return self._now_cash

    def generate_report(self):
        start_date = self.trading_date[0].date()
        end_date = self.trading_date[-1].date()
        CAGR, SR, MDD, CR = self._calculate_metrics()
        data = {
            "Start Trading Date": [start_date],
            "End Trading Date": [end_date],
            "CAGR": [f"{CAGR:.3f}"],
            "Sharpe Ratio": [f"{SR:.3f}"],
            "Maximum Drawdown": [f"{MDD:.3f}"],
            "Calmar Ratio": [f"{CR:.3f}"]
        }
        title = "Trading Performance Metrics"
        results = pd.DataFrame(data)
        results.to_csv("simulation_trading_results.csv")
        date = list(map(lambda dt: dt.date(), self.trading_date))
        asset_value_record = self.asset_value_history
        data = {
            "Asset Value": asset_value_record
        }
        record = pd.DataFrame(data, index=date)
        record.index.name = 'Date'
        record.to_csv("asset_value_record.csv")
        table = [
            ["Start Trading Date", date[0]],
            ["End Trading Date", date[-1]],
            ["CAGR", f"{CAGR:.3f}"],
            ["Sharpe Ratio", f"{SR:.3f}"],
            ["Max Drawdown", f"{MDD:.3f}"],
            ["Calmar Ratio", f"{CR:.3f}"]
        ]

        print(title)
        print("=" * len(title))
        # print(tabulate(results, headers="keys", tablefmt="grid"))
        print(tabulate(table, headers=["Metric", "Value"], tablefmt="grid"))

    def _calculate_metrics(self):
        wealth = np.array(self.asset_value_history)
        assert len(wealth.shape) == 1
        trade_ror = wealth[1:] / wealth[:-1] - 1
        years = len(wealth) / 252
        CAGR = (wealth[-1] / wealth[0]) ** np.sqrt(1 / years)

        APR = np.mean(trade_ror) * 252
        AVOL = np.std(trade_ror, ddof=1) * np.sqrt(252)
        SR = APR / AVOL
        drawdown = (np.maximum.accumulate(wealth) - wealth) / np.maximum.accumulate(wealth)
        MDD = np.max(drawdown, axis=-1)
        CR = APR / MDD
        return CAGR, SR, MDD, CR
