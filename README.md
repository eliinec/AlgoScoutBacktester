# AlgoScoutBacktester
A Python-based backtesting tool that tests a moving average crossover strategy agasint real stock market data.

## Features: 
- Downloads historical price data for any ticker using yfinance
- Generates buy/sell signals using a 5day and 20day moving average crossover with a volatility filter
- Simulates a portfolio and compares performance against buy & hold
- Reports Sharpe Ratio, total return, max drawdown, and directional accuracy
  
## How to use:
1. Clone the repository
2. Install: pip install -r requirements.txt
3. Run python algo_scout.py
