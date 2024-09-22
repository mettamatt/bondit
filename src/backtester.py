# src/backtester.py

import backtrader as bt  # type: ignore
import pandas as pd
import yfinance as yf  # type: ignore


# Define a Buy-and-Hold Strategy with unique names for each ETF
class BuyAndHold(bt.Strategy):
    def __init__(self) -> None:
        self.buy_signal = False  # Ensure we buy only once

    def next(self) -> None:
        if not self.position and not self.buy_signal:  # No position and no prior buy
            self.buy()  # Buy and hold
            self.buy_signal = True


# Fetch historical data for an ETF from yfinance
def fetch_data(symbol: str, start_year: int, end_year: int) -> pd.DataFrame:
    # Convert start and end years to full date strings (Jan 1 to Dec 31)
    start = f"{start_year}-01-01"
    end = f"{end_year}-12-31"

    # Fetch data using yfinance
    data: pd.DataFrame = yf.download(symbol, start, end)
    return data


# Function to run the backtest
def run_backtest(start_year: int, end_year: int) -> None:
    # Store final portfolio values for each ETF in a dictionary
    final_values = {}

    # List of ETF symbols
    symbols = ["BND", "VTIP", "BIV", "SPY"]

    for symbol in symbols:
        # Initialize a separate Cerebro instance for each ETF
        cerebro = bt.Cerebro()

        # Fetch data for the ETF
        data_df = fetch_data(symbol, start_year, end_year)
        data_feed = bt.feeds.PandasData(dataname=data_df)
        cerebro.adddata(data_feed, name=symbol)  # Add each data feed to cerebro

        # Add the BuyAndHold strategy
        cerebro.addstrategy(BuyAndHold)

        # Set initial cash (same for each strategy, but separate per cerebro instance)
        cerebro.broker.setcash(100000.0)

        # Set commission (assuming the same commission for all ETFs)
        cerebro.broker.setcommission(commission=0.0001)  # 0.01% commission

        # Run the backtest
        cerebro.run()

        # Store the final portfolio value for each ETF
        final_value = (
            cerebro.broker.getvalue()
        )  # Get current value (includes cash and holdings)
        final_values[symbol] = final_value
        print(f"Ending Portfolio Value for {symbol}: {final_value:.2f}")

    # Determine which ETF performed the best (had the highest final value)
    winner = max(final_values, key=lambda symbol: final_values[symbol])
    print(
        f"The ETF with the highest portfolio value is {winner} with a value of {final_values[winner]:.2f}"
    )


# Run the backtest for a given range of years
run_backtest(start_year=2013, end_year=2023)
