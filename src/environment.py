import gym
from gym import spaces
import numpy as np
import pandas as pd


class StockTradingEnv(gym.Env):
    def __init__(self, df, window_size=10, initial_balance=10000):
        super().__init__()
        self.df = df.select_dtypes(include=np.number).dropna().copy()
        self._ensure_close_column()
        self.window_size = window_size
        self.initial_balance = initial_balance

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(window_size, self.df.shape[1]),
            dtype=np.float32
        )
        self.reset()

    def _ensure_close_column(self):
        if 'Close' in self.df.columns:
            return
        close_candidates = [
            col for col in self.df.columns
            if isinstance(col, str) and col.lower().startswith('close')
        ]
        if not close_candidates:
            raise KeyError("No column named 'Close' or starting with 'Close' found.")
        self.df.rename(columns={close_candidates[0]: 'Close'}, inplace=True)

    def reset(self):
        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.shares = 0
        self.net_worth = self.initial_balance
        self.net_worth_history = [self.initial_balance]
        return self._get_obs()

    def _get_obs(self):
        frame = self.df.iloc[self.current_step - self.window_size:self.current_step]
        return frame.values.astype(np.float32)

    def step(self, action):
        self.current_step += 1
        done = self.current_step >= len(self.df)

        prev_net_worth = self.net_worth
        price = self.df['Close'].iloc[self.current_step - 1].item()

        if action == 1 and self.balance > price:
            shares_to_buy = self.balance // price
            self.shares += shares_to_buy
            self.balance -= shares_to_buy * price
        elif action == 2 and self.shares > 0:
            self.balance += self.shares * price
            self.shares = 0

        self.net_worth = self.balance + self.shares * price
        self.net_worth_history.append(self.net_worth)

        reward = self.net_worth - prev_net_worth
        obs = self._get_obs() if not done else np.zeros(self.observation_space.shape)

        return obs, reward, done, {'net_worth_history': self.net_worth_history}