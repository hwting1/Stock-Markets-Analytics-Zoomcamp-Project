import numpy as np
import json
import torch
from utils.agent import Agent
from utils.brokerapi import StockBrokerAPI
from utils.transformer import DataTransformer

with open("train_config.json", "r") as f:
    config = json.load(f)

broker = StockBrokerAPI(tickers=config["tickers"])
transformer = DataTransformer(tickers=config["tickers"], seq_len=config["seq_len"])
train_data = broker.get_data(start_date=config["train_start"], end_date=config["train_end"])
transformer.transform(train_data, prefix="train", save=True)

train_data = torch.from_numpy(np.load("data/train_data.npy")).float()
train_trading_prices = torch.from_numpy(np.load("data/train_trading_prices.npy")).float()
train_dataset = torch.utils.data.TensorDataset(train_data, train_trading_prices)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)

if config["valid_start"] and config["valid_end"]:
    valid_data = broker.get_data(start_date=config["valid_start"], end_date=config["valid_end"])
    transformer.transform(valid_data, prefix="valid", save=True)
    valid_data = torch.from_numpy(np.load("data/valid_data.npy")).float()
    valid_trading_prices = torch.from_numpy(np.load("data/valid_trading_prices.npy")).float()
    valid_dataset = torch.utils.data.TensorDataset(valid_data, valid_trading_prices)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=1)
else:
    valid_dataloader = None

agent = Agent(device=config["device"], hidden_dim=config["hidden_dim"], seq_len=config["seq_len"], asset_num=config["asset_num"],
              feature_num=config["feature_num"], bias=config["bias"], dropout=config["dropout"], lstm_num=config["lstm_num"],
              init_money=config["init_money"], fee=config["fee"])

agent.learn(lr=config["lr"], steps=config["steps"], epochs_per_step=config["epochs_per_step"], gamma=config["gamma"],
            head_mask=config["head_mask"], tail_mask=config["tail_mask"], reward_scaling=config["reward_scaling"],
            dataloader=train_dataloader, config=config, val_dataloader=valid_dataloader, model_path=config["model_path"])