import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def fetch_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end)
    return df

def apply_strategy(df):
    df['Fast_MA'] = df['Close'].rolling(window=5).mean()
    df['Slow_MA'] = df['Close'].rolling(window=20).mean()
    df['Signal'] = np.where(df['Fast_MA'] > df['Slow_MA'], 1, 0)
    df['Crossover'] = df['Signal'].diff()
    return df

def calculate_performance(df, cash):
    df['Daily_Pct'] = df['Close'].pct_change()
    df['Strat_Pct'] = df['Daily_Pct'] * df['Signal'].shift(1)
    
    df['Equity_Base'] = (1 + df['Daily_Pct']).fillna(1).cumprod() * cash
    df['Equity_Strat'] = (1 + df['Strat_Pct']).fillna(1).cumprod() * cash
    
    df['Peak'] = df['Equity_Strat'].cummax()
    df['Drawdown'] = (df['Equity_Strat'] - df['Peak']) / df['Peak']
    
    return df

def predict_future(df):
    y = df['Close'].tail(30).values
    X = np.arange(len(y)).reshape(-1, 1)
    model = LinearRegression().fit(X, y)
    future_X = np.arange(len(y), len(y) + 5).reshape(-1, 1)
    return model.predict(future_X)

def visualize_all(df, ticker, prediction):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=False)

    ax1.plot(df.index, df['Close'], label='Price', color='black', alpha=0.3)
    ax1.plot(df.index, df['Fast_MA'], label='5-Day MA', color='blue')
    ax1.plot(df.index, df['Slow_MA'], label='20-Day MA', color='orange')
    
    buys = df[df['Crossover'] == 1]
    sells = df[df['Crossover'] == -1]
    ax1.scatter(buys.index, buys['Close'], marker='^', color='green', s=100)
    ax1.scatter(sells.index, sells['Close'], marker='v', color='red', s=100)
    ax1.set_title(f"{ticker} Technical Strategy")
    ax1.legend()

    ax2.plot(df.index, df['Equity_Base'], label='Buy & Hold', color='gray', linestyle='--')
    ax2.plot(df.index, df['Equity_Strat'], label='Algo Strategy', color='purple', linewidth=2)
    ax2.set_title("Wealth Growth Comparison")
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

def main():
    ticker = input("Ticker: ").upper()
    start = input("Start (YYYY-MM-DD): ")
    end = input("End (YYYY-MM-DD): ")
    investment = float(input("Investment Amount: "))

    data = fetch_data(ticker, start, end)
    data = apply_strategy(data)
    data = calculate_performance(data, investment)
    
    forecast = predict_future(data)
    
    print(f"\nFinal Portfolio (Base): ${data['Equity_Base'].iloc[-1]:.2f}")
    print(f"Final Portfolio (Algo): ${data['Equity_Strat'].iloc[-1]:.2f}")
    print(f"Max Drawdown: {data['Drawdown'].min() * 100:.2f}%")
    print(f"Next 5-Day Forecast: {forecast}")

    visualize_all(data, ticker, forecast)

if __name__ == "__main__":
    main()