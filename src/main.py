import pandas as pd
from src.data_fetcher import fetch_data
from src.indicators import add_indicators
from src.environment import StockTradingEnv
from src.dqn_agent import DQNAgent
from tqdm import tqdm
import numpy as np


def train_agent(symbol, episodes=50, model_path='dqn_model.pth', progress_callback=None):
    # 1. Prepare Data
    df = fetch_data(symbol)
    df = add_indicators(df)
    df.dropna(inplace=True)

    # 2. Setup Environment
    window_size = 10
    env = StockTradingEnv(df, window_size=window_size)
    state_dim = env.observation_space.shape[0] * env.observation_space.shape[1]
    action_dim = env.action_space.n

    # 3. Setup Agent
    agent = DQNAgent(state_dim, action_dim)
    batch_size = 32

    # 4. Training Loop
    for e in range(episodes):
        state = env.reset()
        state = state.flatten()

        num_steps = len(df) - 1 - env.current_step
        pbar = tqdm(range(env.current_step, len(df) - 1), desc=f"Episode {e + 1}/{episodes}")

        for i, time in enumerate(pbar):
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            next_state = next_state.flatten()

            agent.memorize(state, action, reward, next_state, done)
            state = next_state

            if done:
                break

            if len(agent.memory) > batch_size:
                loss = agent.replay(batch_size)
                pbar.set_postfix(loss=f"{loss:.4f}")

        agent.update_target_model()
        print(f"Episode {e + 1}/{episodes}, Net Worth: {env.net_worth:.2f}, Epsilon: {agent.epsilon:.2f}")
        if progress_callback:
            progress_callback((e + 1) / episodes)

    # 5. Save Model
    agent.save(model_path)
    print(f"Model saved to {model_path}")


if __name__ == '__main__':
    train_agent(symbol="AAPL", episodes=10)
