# Final Project for 'Stock-Markets-Analytics-Zoomcamp'

## Introduction
High-frequency trading (HFT) is a complex financial task treated as a near real-time sequential decision problem. Traditional approaches often forecast equity trends and optimize weights via combinatorial optimization. However, these methods face challenges such as computational inefficiencies and limitations in handling a large number of assets with discrete action spaces.

An efficient DRL-based policy optimization (DRPO) method for HFT has been proposed to address these issues. This method models portfolio management as a Markov Decision Process, directly inferring equity weights to maximize accumulated returns. The environment is separated into "static" market states and "dynamic" portfolio weight states, simplifying agent interactions without losing interpretability. A reward expectation calculation algorithm using probabilistic dynamic programming enables agents to collect feedback without complex trajectory sampling.

## Reference
1. Han, L., Ding, N., Wang, G., Cheng, D., & Liang, Y. (2023, August). Efficient Continuous Space Policy Optimization for High-frequency Trading. In Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (pp. 4112-4122). [Source code](https://github.com/finint/DRPO)


## Dependencies
- torch
- numpy<2
- pandas
- python-dateutil
- yfinance
- tabulate
- wandb (optional, for using validation data during training)

### Environment Setting
Set up a Python virtual environment and install the dependencies:
```bash
pip install -r requirements.txt
```

## Training Model and Simulation Trading

### Training
- Ensure your computer has an NVIDIA GPU to train your model.
- You can modify `train_config.json` to customize your training configuration.
- Run ```python train.py``` to start the training process.

### Simulation Trading
- Make sure you have at least one pre-trained model file in the `model` folder (I have provided mine).
- You can modify `trade_config.json` to customize your trading configuration.
- Run ```python trade.py``` to simulate real-time trading.

## Dataset

| Market | Num. of stocks | Train   | Validation | Test    | features                                        |
|--------|----------------|------------|------------|---------|-------------------------------------------------|
| DIJA 30| 30             | 2001-2021  | 2022       | 2023| open, close, high, low prices, true range ratio |

- OCHL prices are normalized by dividing them by the previous day's closing price and then taking the logarithm.
-  "DOW," "CRM," and "V" have been replaced by "XOM," "PFE," and "RTX" due to insufficient data.

## Results

| Start Date | End Date   | CAGR  | Sharpe Ratio | Maximum Drawdown | Calmar Ratio |
|------------|------------|-------|--------------|------------------|--------------|
| 2023-01-03 | 2023-12-29 | 1.204 | 1.223        | 0.154            | 1.304        |

  