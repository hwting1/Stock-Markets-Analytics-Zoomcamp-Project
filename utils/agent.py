import torch
from torch import nn
import numpy as np
import wandb

class Actor(nn.Module):

    def __init__(self, hidden_dim, seq_len, asset_num, feature_num, bias, dropout, lstm_num):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.asset_num = asset_num
        self.feature_num = feature_num
        self.lstm_num = lstm_num
        self.bias = bias
        self.dropout = dropout

        self.LSTM = nn.LSTM(
            input_size = self.feature_num + 1,
            hidden_size = self.hidden_dim,
            num_layers = self.lstm_num,
            bias = self.bias,
            batch_first = True,
            dropout = self.dropout,
        )

        self.MLP = nn.Sequential(
            nn.Linear(self.hidden_dim,
                      self.hidden_dim * 7, bias=self.bias),
            nn.SiLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim * 7, 128, bias=self.bias),
            nn.SiLU(),
            nn.Dropout(self.dropout),
            nn.Linear(128, 64, bias=self.bias),
            nn.SiLU(),
            nn.Dropout(self.dropout),
            nn.Linear(64, 1, bias=self.bias),
        )

    def forward(self, inputs, hx, cx):
        output, (hx, cx) = self.LSTM(inputs, (hx, cx))
        output = output.squeeze(dim=1)
        output = self.MLP(output)
        return output, hx, cx


class Agent:

    def __init__(self, device, hidden_dim, seq_len, asset_num, feature_num, bias, dropout, lstm_num,
                 init_money, fee):

        self.device = device
        self.policy = Actor(hidden_dim, seq_len, asset_num, feature_num, bias, dropout, lstm_num)
        self.policy.to(self.device)
        self.lstm_num = lstm_num
        self.hidden_dim = hidden_dim
        self.init_money = init_money
        self.fee = fee

        for name, param in self.policy.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(param)

    def learn(self, lr, steps, epochs_per_step, gamma, head_mask, tail_mask, reward_scaling,
              dataloader, config, val_dataloader=None, model_path=None):
        if val_dataloader:
            wandb.login()
            run = wandb.init(project='Stock-Markets-Analytics-Zoomcamp-Project', config=config)
        optimizer = torch.optim.AdamW(self.policy.parameters(), lr=lr)

        for i in range(steps+1):
            difficulty = i / steps
            for _ in range(epochs_per_step):
                optimizer = self.learn_one_epoch(dataloader, difficulty, optimizer, gamma, head_mask, tail_mask, reward_scaling)

            optimizer.step()
            optimizer.zero_grad()
            if val_dataloader:
                val_result = self.evaluate(val_dataloader)
                wandb.log(val_result)
        if model_path:
            self.save_model(model_path)

    def learn_one_epoch(self, dataloader, difficulty, optimizer, gamma, head_mask, tail_mask, reward_scaling):

        self.policy.train()
        for states, trading_prices in dataloader:
            states = states.to(self.device)
            trading_prices = trading_prices.to(self.device)

            bs, ts, sn, fn = states.shape
            fee = self.fee * difficulty
            out_time_stock = []
            state = torch.zeros(bs, ts+1, sn, device=self.device)
            new_state = torch.zeros(bs, ts+1, sn, device=self.device)
            cash = torch.zeros(bs, ts+1, device=self.device)
            hold = torch.zeros(bs, ts+1, device=self.device)
            asset = torch.zeros(bs, ts+1, device=self.device)
            cash[:, 0] = self.init_money
            asset[:, 0] = cash[:, 0]
            hx = torch.ones(self.lstm_num, bs* sn, self.hidden_dim,
                            device=self.device)
            cx = torch.ones(self.lstm_num, bs* sn, self.hidden_dim,
                            device=self.device)
            obs_omega = torch.softmax(state[:, 0, :], dim=-1)
            # obs_omega = state[:, 0, :]

            for i in range(ts):
                inputs = states[:, i:i+1, :, :]
                inputs = inputs.reshape(-1, fn)
                input_temp = torch.concat([inputs, obs_omega.reshape(-1,1)], dim=1)
                input_temp = input_temp.reshape(bs*sn, 1, -1)
                output, hx, cx = self.policy(input_temp, hx, cx)
                output = output.reshape(bs, sn)
                out_time_stock.append(output)

                with torch.no_grad():
                    this_state = state[:, i, :]
                    trading_price = trading_prices[:, i, :]
                    this_cash = cash[:, i-1]
                    sell_actions, buy_actions, next_cash = self.action_convertor(output, this_state, fee, trading_price, this_cash)
                    weight_change = sell_actions + buy_actions
                    next_state = this_state + weight_change
                    this_hold = torch.sum( next_state * trading_price, dim=-1)
                    state[:, i+1, :] = next_state
                    cash[:, i+1] = next_cash
                    hold[:, i+1] = this_hold
                    asset[:, i+1] = next_cash + this_hold
                    obs_omega = torch.softmax(next_state, dim=-1)

        out_time_stock = torch.stack(out_time_stock, dim=1)
        with torch.no_grad():
            new_state[:, :-1, :] = state[:, 1:, :]
            state_change = (new_state-state)[:, :-1, :]
            asset = asset / asset[:, 0].unsqueeze(1).repeat_interleave(ts+1, dim=1) # (bs, 141, 30)
            new_asset = torch.zeros_like(asset, device=self.device)
            new_asset[:, :-1] = asset[:, 1:]
            rewards = (new_asset-asset)[:, :-1]
            rewards_pct = rewards / asset[:, 0].unsqueeze(1).repeat_interleave(ts, dim=1)

            bs, ts, sn = state_change.shape
            E_pi = torch.zeros(bs, ts).to(self.device)
            final_price = asset[:, -1]
            initial_price = asset[:, 0]
            E_pi[:, -1] = final_price / initial_price
            for i in range(ts-2, -1, -1):
                E_pi[:, i] = rewards_pct[:, i] + gamma * E_pi[:, i+1]

            rewards_pct = rewards_pct[:, head_mask:tail_mask]

            E_pi = E_pi[:, head_mask+1:tail_mask+1]
            V_pi = rewards_pct + gamma * E_pi
            V_pi = V_pi * reward_scaling
            V_pi = V_pi - torch.mean(V_pi, dim=1, keepdim=True)
            V_pi = V_pi.unsqueeze(-1).repeat(1, 1, out_time_stock.shape[-1])

        loss = -torch.mean(V_pi * out_time_stock[:, head_mask:tail_mask, :])
        # optimizer.zero_grad()
        loss.backward()
        # optimizer.step()
        return optimizer

    @torch.no_grad()
    def evaluate(self, dataloader):
        fee = self.fee
        self.policy.eval()
        for states, trading_prices in dataloader:
            states = states.to(self.device)
            trading_prices = trading_prices.to(self.device)

            bs, ts, sn, fn = states.shape
            state = torch.zeros(bs, ts+1, sn, device=self.device)
            cash = torch.zeros(bs, ts+1, device=self.device)
            hold = torch.zeros(bs, ts+1, device=self.device)
            asset = torch.zeros(bs, ts+1, device=self.device)
            cash[:, 0] = self.init_money
            asset[:, 0] = cash[:, 0]
            hx = torch.ones(self.lstm_num, bs* sn, self.hidden_dim,
                            device=self.device)
            cx = torch.ones(self.lstm_num, bs* sn, self.hidden_dim,
                            device=self.device)
            obs_omega = torch.softmax(state[:, 0, :], dim=-1)
            # obs_omega = state[:, 0, :]

            for i in range(ts):
                inputs = states[:, i, :, :]
                inputs = inputs.reshape(-1, fn)
                input_temp = torch.concat([inputs, obs_omega.reshape(-1,1)], dim=1)
                input_temp = input_temp.reshape(bs*sn, 1, -1)
                output, hx, cx = self.policy(input_temp, hx, cx)
                output = output.reshape(bs, sn)

                this_state = state[:, i, :]
                trading_price = trading_prices[:, i, :]
                this_cash = cash[:, i]
                sell_actions, buy_actions, next_cash = self.action_convertor(output, this_state, fee, trading_price, this_cash)
                weight_change = sell_actions + buy_actions
                next_state = this_state + weight_change
                this_hold = torch.sum(next_state * trading_price, dim=-1)
                state[:, i+1, :] = next_state
                cash[:, i+1] = next_cash
                hold[:, i+1] = this_hold
                asset[:, i] = next_cash + this_hold
                obs_omega = torch.softmax(next_state, dim=-1)

            asset[:, -1] = cash[:, -1] + torch.sum(state[:, -1, :].squeeze() * trading_prices[:, -1, :].squeeze(), axis=-1)
            asset[:, 0] = self.init_money

        asset = np.squeeze(asset.cpu().numpy())
        asset = asset / asset[0]
        APV = asset[-1] / asset[0]
        trade_ror = asset[1:] / asset[:-1] - 1
        annualized_return = np.mean(trade_ror) * 252
        annualized_std = np.std(trade_ror, ddof=1) * np.sqrt(252)
        sr = annualized_return / annualized_std
        drawdown = (np.maximum.accumulate(asset) - asset) / np.maximum.accumulate(asset)
        mdd = np.max(drawdown, axis=-1)
        cr = annualized_return / mdd

        valid_result = {'valid_APV': APV,
                        'valid_SR': sr,
                        'valid_MDD': mdd,
                        'valid_CR': cr}
        return valid_result

    @torch.no_grad()
    def trade(self, inputs, hx, cx, this_state, trading_price, this_cash):
        self.policy.eval()
        inputs = torch.from_numpy(inputs).to(self.device).float()
        hold_shares = torch.from_numpy(this_state).to(self.device).float()
        hold_shares = hold_shares.reshape(-1, 1, 1)
        obs_omega = torch.softmax(hold_shares, dim=0)
        inputs = torch.concat([inputs, obs_omega], dim=-1)
        trading_price = torch.from_numpy(trading_price).to(self.device).float()
        this_state = torch.from_numpy(this_state).to(self.device).float()
        output, hx, cx = self.policy(inputs, hx, cx)
        output = output.reshape(1,-1)
        sell_actions, buy_actions, next_cash = self.action_convertor(output, this_state, self.fee, trading_price, this_cash)
        return sell_actions, buy_actions, next_cash, hx, cx

    def action_convertor(self, output, this_state, fee, trading_price, this_cash):
        output = output - torch.mean(output, dim =-1, keepdim=True)
        next_state = this_state + output
        zero_actions = torch.zeros_like(next_state).to(self.device)
        next_state = torch.where(next_state <= 0, zero_actions, next_state)
        weight_change = next_state - this_state
        zero_actions = torch.zeros_like(weight_change).to(self.device)
        sell_actions = torch.where(weight_change >= 0, zero_actions, weight_change)
        buy_actions = torch.where(weight_change <= 0, zero_actions, weight_change)

        buy_costs = buy_actions * trading_price * fee
        sell_costs = sell_actions * trading_price * -fee

        sell = torch.sum((-1 * sell_actions * trading_price - sell_costs), dim=-1)
        this_cash = this_cash + sell
        buy = torch.sum((buy_actions * trading_price + buy_costs), dim=-1)

        # Mandatory use of all cash
        factor = this_cash / buy
        temp_factor = torch.zeros_like(factor)
        factor = torch.where(torch.isnan(factor), temp_factor, factor)
        factor = torch.where(torch.isinf(factor), temp_factor, factor)
        sn = buy_actions.shape[-1]
        buy_actions = buy_actions * factor.unsqueeze(-1).repeat_interleave(sn, dim=-1)
        buy = buy * factor
        next_cash = this_cash - buy
        return sell_actions, buy_actions, next_cash

    def save_model(self, path):
        torch.save(self.policy.state_dict(), path)

    def load_model(self, path):
        self.policy.load_state_dict(torch.load(path))
