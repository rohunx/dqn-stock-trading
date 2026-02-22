import torch
from src.data_fetcher import fetch_data
from src.indicators import add_indicators
from src.environment import StockTradingEnv
from src.dqn_agent import DQNAgent
import numpy as np
import pandas as pd


def backtest_agent(symbol, model_path='dqn_model.pth'):
    df = fetch_data(symbol)
    df = add_indicators(df)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(map(str, col)).strip() for col in df.columns]

    df.dropna(inplace=True)
    df_eval = df.copy()

    window_size = 10
    env = StockTradingEnv(df, window_size=window_size)
    state_dim = env.observation_space.shape[0] * env.observation_space.shape[1]
    action_dim = env.action_space.n

    agent = DQNAgent(state_dim, action_dim, epsilon=0.0)
    agent.load(model_path)

    state = env.reset().flatten()
    done = False
    initial_net_worth = env.initial_balance
    buy_indicators = []
    sell_indicators = []

    while not done:
        action = agent.act(state)
        current_indicators = df_eval.iloc[env.current_step]

        if action == 1:
            buy_indicators.append(current_indicators)
        elif action == 2:
            sell_indicators.append(current_indicators)

        next_state, reward, done, info = env.step(action)
        state = next_state.flatten()

    print(f"Backtest Complete. Final Net Worth: {env.net_worth:.2f}")

    net_worth_history = info['net_worth_history']
    dates = df.index[env.window_size - 1: env.window_size - 1 + len(net_worth_history)]

    buy_df = pd.DataFrame(buy_indicators)
    sell_df = pd.DataFrame(sell_indicators)

    indicator_cols = [col for col in ['RSI', 'MACD', 'SMA_20', 'SMA_50'] if col in df.columns]
    avg_buy_indicators = buy_df[indicator_cols].mean().to_dict() if not buy_df.empty else {}
    avg_sell_indicators = sell_df[indicator_cols].mean().to_dict() if not sell_df.empty else {}

    stats = {
        "initial_net_worth": initial_net_worth,
        "final_net_worth": env.net_worth,
        "net_worth_history": net_worth_history,
        "dates": dates,
        "total_buys": len(buy_df),
        "total_sells": len(sell_df),
        "avg_buy_indicators": avg_buy_indicators,
        "avg_sell_indicators": avg_sell_indicators
    }

    return stats


if __name__ == '__main__':
    backtest_agent(symbol="AAPL")
