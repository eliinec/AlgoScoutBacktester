import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def fetch_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end)
    df.columns = df.columns.get_level_values(0)  
    return df


def apply_strategy(df):
    df['Fast_MA'] = df['Close'].rolling(5).mean()
    df['Slow_MA'] = df['Close'].rolling(20).mean()
    
    df['Volatility'] = df['Close'].pct_change().rolling(10).std()
    
    df['Signal'] = np.where(
        (df['Fast_MA'] > df['Slow_MA']) &
        (df['Volatility'] < df['Volatility'].median()),
        1,
        0
    )
    
    df['Crossover'] = df['Signal'].diff()
    return df


def rolling_predictions(df, window=30, horizon=5):
    predictions = [np.nan] * len(df)

    for i in range(window, len(df) - horizon):
        y = df['Close'].iloc[i-window:i].values
        X = np.arange(window).reshape(-1, 1)

        model = LinearRegression().fit(X, y)

        future_index = window + horizon - 1
        pred = model.predict([[future_index]])[0]

        predictions[i] = pred

    df['Prediction'] = predictions
    return df


def calculate_performance(df, cash):
    df['Daily_Pct'] = df['Close'].pct_change()
    df['Strat_Pct'] = df['Daily_Pct'] * df['Signal'].shift(1)

    
    cost_per_trade = 0.001
    df['Trades'] = df['Signal'].diff().abs()
    df['Cost'] = df['Trades'] * cost_per_trade
    df['Strat_Pct'] = df['Strat_Pct'] - df['Cost']

   
    df['Equity_Base'] = (1 + df['Daily_Pct']).fillna(1).cumprod() * cash
    df['Equity_Strat'] = (1 + df['Strat_Pct']).fillna(1).cumprod() * cash

   
    df['Peak'] = df['Equity_Strat'].cummax()
    df['Drawdown'] = (df['Equity_Strat'] - df['Peak']) / df['Peak']

    
    df['Peak_Base'] = df['Equity_Base'].cummax()
    df['Drawdown_Base'] = (df['Equity_Base'] - df['Peak_Base']) / df['Peak_Base']

    return df

def performance_metrics(df):
    returns = df['Strat_Pct'].dropna()

    sharpe = np.sqrt(252) * returns.mean() / returns.std()
    total_return = df['Equity_Strat'].iloc[-1] / df['Equity_Strat'].iloc[0] - 1
    max_dd = df['Drawdown'].min()
    max_dd_base = df['Drawdown_Base'].min() 

    print(f"\nSharpe Ratio: {sharpe:.2f}")
    print(f"Total Return: {total_return:.2%}")
    print(f"Max Drawdown: {max_dd:.2%}")
    print(f"Max Drawdown (Buy & Hold): {max_dd_base:.2%}")  

def prediction_accuracy(df):
    if 'Prediction' not in df.columns:
        print("Prediction column missing.")
        return

    df['Future_Price'] = df['Close'].shift(-5)

    valid = df.dropna(subset=['Prediction', 'Future_Price'])

    if len(valid) == 0:
        print("No valid prediction data.")
        return

    correct = (
        (valid['Prediction'] > valid['Close']) ==
        (valid['Future_Price'] > valid['Close'])
    )

    accuracy = correct.mean()
    print(f"Directional Accuracy: {accuracy:.2%}")



def visualize(df, ticker):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))


    ax1.plot(df.index, df['Close'], label='Price', alpha=0.5)
    ax1.plot(df.index, df['Fast_MA'], label='5-Day MA')
    ax1.plot(df.index, df['Slow_MA'], label='20-Day MA')

    buys = df[df['Crossover'] == 1]
    sells = df[df['Crossover'] == -1]

    ax1.scatter(buys.index, buys['Close'], marker='^')
    ax1.scatter(sells.index, sells['Close'], marker='v')

    ax1.set_title(f"{ticker} Strategy")
    ax1.legend()


    ax2.plot(df.index, df['Equity_Base'], label='Buy & Hold')
    ax2.plot(df.index, df['Equity_Strat'], label='Strategy')
    ax2.set_title("Portfolio Value")
    ax2.legend()


    ax3.plot(df.index, df['Drawdown_Base'], label='Buy & Hold Drawdown', color='steelblue')
    ax3.plot(df.index, df['Drawdown'], label='Strategy Drawdown', color='orange')
    ax3.set_title("Drawdown Comparison")
    ax3.legend()

    plt.tight_layout()
    plt.show()


def main():
    ticker = input("Ticker: ").upper()
    start = input("Start (YYYY-MM-DD): ")
    end = input("End (YYYY-MM-DD): ")
    investment = float(input("Investment Amount: "))

    data = fetch_data(ticker, start, end)

    data = apply_strategy(data)
    data = rolling_predictions(data)
    data = calculate_performance(data, investment)

    print(f"\nFinal Portfolio (Base): ${data['Equity_Base'].iloc[-1]:.2f}")
    print(f"Final Portfolio (Strategy): ${data['Equity_Strat'].iloc[-1]:.2f}")

    performance_metrics(data)
    prediction_accuracy(data)

    visualize(data, ticker)


if __name__ == "__main__":
    main()