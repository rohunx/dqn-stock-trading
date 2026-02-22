# 📈 Deep Q-Network (DQN) Stock Trading Strategy

A quantitative trading strategy implementing **Deep Reinforcement Learning (DQN)** for stock market strategy development, backtesting, and interactive analysis.

Built with:

- 🧠 PyTorch (Deep Q-Network RL)  
- 📊 Technical Indicators (SMA, RSI, MACD)  
- 📈 Historical Backtesting Engine  
- 🖥 Streamlit Interactive Dashboard   
- 🧪 Modular, Research-Oriented Architecture  

---

## 🚀 Overview

This project simulates and evaluates reinforcement learning–based trading strategies using historical market data.

The system includes:

- Historical data ingestion (Yahoo Finance via `yfinance`)
- Feature engineering with common technical indicators
- Deep Q-Network (DQN) agent
- Backtesting with performance metrics
- Interactive Streamlit frontend for symbol analysis

Designed to demonstrate:

- Reinforcement learning engineering  
- Quantitative finance modeling  
- Clean, scalable software architecture  
- Modular ML system design  

---

## 🏗 Project Structure

```text
dqn-stock-trading/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   └── historical/          # Cached historical datasets
│
├── notebooks/
│   └── analysis.ipynb       # Exploratory data analysis & research
│
├── src/
│   ├── data_fetcher.py      # Market data ingestion
│   ├── indicators.py        # Feature engineering (SMA, RSI, MACD)
│   ├── environment.py       # Gym-style trading environment
│   ├── dqn_agent.py         # Deep Q-Network implementation (PyTorch)
│   ├── backtest.py          # Portfolio simulation engine
│   ├── visualization.py     # Charting utilities
│   ├── chatbot.py           # AI strategy analysis assistant
│   └── main.py              # CLI experimentation entrypoint
│
└── frontend/
    └── streamlit_app.py     # Interactive dashboard

