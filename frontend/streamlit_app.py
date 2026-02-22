import streamlit as st
from src.data_fetcher import fetch_data
from src.indicators import add_indicators
from src.main import train_agent
from src.backtest import backtest_agent
import matplotlib.pyplot as plt
import pandas as pd
import os

st.title("📈 Stock Market DQN Trading Dashboard")

symbol = st.text_input("Enter Stock Symbol (e.g., AAPL)", "AAPL")
model_path = f"dqn_{symbol}.pth"

if st.button("Fetch & Analyze"):
    data_load_state = st.text('Loading data...')
    df = fetch_data(symbol)
    df = add_indicators(df)
    data_load_state.text('Loading data... done!')

    st.subheader(f"{symbol} Raw Data")
    st.write(df.tail())

    # Plot indicators
    st.subheader(f"{symbol} Price & Indicators")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df['Close'], label='Close Price')
    if 'SMA_20' in df.columns:
        ax.plot(df['SMA_20'], label='SMA 20')
    if 'SMA_50' in df.columns:
        ax.plot(df['SMA_50'], label='SMA 50')
    ax.set_title(f"{symbol} Price & SMA")
    ax.legend()
    st.pyplot(fig)

st.sidebar.subheader("DQN Agent")
episodes = st.sidebar.slider("Training Episodes", 1, 100, 10)

if st.sidebar.button("Train & Evaluate Agent"):
    if not os.path.exists(model_path):
        progress_bar = st.sidebar.progress(0)
        status_text = st.sidebar.empty()
        with st.spinner(f"Training agent on {symbol} for {episodes} episodes..."):
            train_agent(symbol, episodes=episodes, model_path=model_path,
                        progress_callback=lambda p: progress_bar.progress(p))
        status_text.success("Training complete!")
        progress_bar.empty()
    else:
        st.info(f"Found existing model: {model_path}. Skipping training.")

    with st.spinner("Running backtest..."):
        stats = backtest_agent(symbol, model_path=model_path)
    st.success("Backtest complete!")

    st.subheader("DQN Agent Performance")

    col1, col2 = st.columns(2)
    col1.metric("Starting Net Worth", f"${stats['initial_net_worth']:,.2f}")
    col2.metric("Final Net Worth", f"${stats['final_net_worth']:,.2f}",
                f"{stats['final_net_worth'] - stats['initial_net_worth']:,.2f}")

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(stats['dates'], stats['net_worth_history'], label='Agent Net Worth')
    ax.set_title("Agent Net Worth Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Net Worth ($)")
    ax.legend()
    st.pyplot(fig)

    st.subheader("Trading Statistics")
    st.write(f"Total Buy Actions: {stats['total_buys']}")
    st.write(f"Total Sell Actions: {stats['total_sells']}")

    if stats['avg_buy_indicators']:
        st.write("Average Indicator Values on **BUY**:")
        st.json({k: f"{v:.2f}" for k, v in stats['avg_buy_indicators'].items()})

    if stats['avg_sell_indicators']:
        st.write("Average Indicator Values on **SELL**:")
        st.json({k: f"{v:.2f}" for k, v in stats['avg_sell_indicators'].items()})
