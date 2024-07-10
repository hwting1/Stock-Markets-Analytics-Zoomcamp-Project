import numpy as np
import json, time
from dateutil import parser
from datetime import timedelta
import yfinance as yf
import torch
from utils.agent import Agent
from utils.brokerapi import StockBrokerAPI
from utils.transformer import StreamingDataTransformer
from utils.simulator import PortfolioSimulator

with open("trade_config.json", "r") as f:
    config = json.load(f)

start_date = parser.parse(config["start_date"])
end_date = parser.parse(config["end_date"])
broker = StockBrokerAPI(tickers=config["tickers"])
transformer = StreamingDataTransformer(tickers=config["tickers"])
agent = Agent(device=config["device"], hidden_dim=config["hidden_dim"], seq_len=config["seq_len"], asset_num=config["asset_num"],
              feature_num=config["feature_num"], bias=config["bias"], dropout=config["dropout"], lstm_num=config["lstm_num"],
              init_money=config["init_money"], fee=config["fee"])
agent.load_model(config["model_path"])
simulator = PortfolioSimulator(config["init_money"])


def main(start_date, end_date):
    now_date = start_date
    now_cash = simulator.now_cash
    now_cash = torch.tensor(now_cash, dtype=torch.float32).reshape(1).to(config["device"])
    hold_shares = broker.hold_shares
    hold_shares = np.expand_dims(hold_shares, axis=0)
    hx = torch.ones(config["lstm_num"], config["asset_num"], config["hidden_dim"],
                    device=config["device"])
    cx = torch.ones(config["lstm_num"], config["asset_num"], config["hidden_dim"],
                    device=config["device"])
    data = broker.get_data(start_date - timedelta(days=10), start_date - timedelta(days=1))
    transformer.set_pre_close(data)

    while now_date <= end_date:
        check_df = yf.download(tickers="SPY", start=now_date, end=now_date + timedelta(days=1), progress=False)
        if not check_df.empty:
            today_data = broker.get_data(now_date, now_date + timedelta(days=1))
            today_state, today_price = transformer.transform(today_data)
            sell_actions, buy_actions, next_cash, hx, cx = agent.trade(today_state, hx, cx, hold_shares, today_price, now_cash)
            weight_change = sell_actions + buy_actions
            weight_change = weight_change.squeeze().cpu().numpy()
            broker.execute_order(weight_change, now_date)
            hold_shares = broker.hold_shares
            simulator.record(hold_shares, today_price.squeeze(), next_cash.cpu().squeeze().numpy(), now_date)
            hold_shares = np.expand_dims(hold_shares, axis=0)
            now_cash = next_cash
            print(f"Finish trading! Today is {now_date.date()}")

        now_date += timedelta(days=1)
        time.sleep(1)

    simulator.generate_report()

if __name__ == '__main__':
    main(start_date, end_date)