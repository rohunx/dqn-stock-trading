import matplotlib.pyplot as plt

def plot_trading(df):
    plt.figure(figsize=(14,7))
    plt.plot(df['Close'], label='Close Price')
    plt.plot(df['SMA_20'], label='SMA 20')
    plt.plot(df['SMA_50'], label='SMA 50')
    plt.legend()
    plt.show()