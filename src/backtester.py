# src/backtester.py

"""
Simplified Backtesting Script for Bondit Portfolio Management Using Backtrader

This script performs backtesting of the Bondit portfolio allocation strategy and a buy-and-hold strategy for BND from 2011 to 2023.
It utilizes 13 years of historical data (2011-2023) to simulate investment decisions and portfolio rebalancing.

Key Components:
- DataFetcher: Fetches historical economic indicators and ETF price data.
- Portfolio: Manages allocation percentages and enforces constraints.
- DecisionEngine: Applies decision rules based on economic indicators to adjust allocations.
- EconomicIndicator: Processes and analyzes economic data.
- BaseStrategy: Encapsulates common strategy functionality.
- BondAllocationStrategy: Strategy for dynamic bond allocation.
- BuyAndHoldBNDStrategy: Implements a simple buy-and-hold strategy for BND.

Usage:
    python backtest.py

Requirements:
    - Python 3.8+
    - pandas
    - backtrader
    - yfinance
    - matplotlib
    - Other dependencies as specified in your project configuration.
"""

import logging
from datetime import datetime
from typing import Any, Dict, Hashable, List, Protocol, Tuple, cast

import backtrader as bt  # type: ignore
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf  # type: ignore

# Import your existing modules
from src.config import INDICATORS
from src.decision_engine import DecisionEngine
from src.indicators import EconomicIndicator
from src.portfolio import Portfolio
from src.utils import (
    collect_economic_indicators,
    initialize_data_storage_and_fetcher,
    initialize_logger,
    load_configuration,
    parse_allocations,
)

# Define the mapping from Vanguard mutual funds to ETFs
VANGUARD_MAPPING: Dict[str, str] = {
    "VBIRX": "BSV",  # Vanguard Short-Term Bond ETF
    "VBLAX": "BND",  # Vanguard Total Bond Market ETF
    "VTAPX": "VTIP",  # Vanguard Short-Term Inflation-Protected Securities ETF
}

# Initialize the logger
logger: logging.Logger = initialize_logger()


def fetch_etf_data(
    symbol: str, start_date: datetime, end_date: datetime
) -> pd.DataFrame:
    """
    Fetch historical price data for a given ETF using yfinance.

    Args:
        symbol (str): The ETF ticker symbol.
        start_date (datetime): The start date.
        end_date (datetime): The end date.

    Returns:
        pd.DataFrame: Historical price data with 'date' as a column.

    Raises:
        ValueError: If no data is fetched for the given ETF.
    """
    logger.info(f"Fetching data for ETF: {symbol}")
    data: pd.DataFrame = yf.download(symbol, start=start_date, end=end_date)
    if data.empty:
        logger.error(f"No data fetched for ETF: {symbol}")
        raise ValueError(f"No data fetched for ETF: {symbol}")
    data.reset_index(inplace=True)
    data.rename(columns={"Date": "date"}, inplace=True)
    return data


def prepare_etf_data(
    symbols: List[str], start_date: datetime, end_date: datetime
) -> Dict[str, bt.feeds.PandasData]:
    data_feeds = {}
    for symbol in symbols:
        try:
            df = fetch_etf_data(symbol, start_date, end_date)
            df.rename(
                columns={
                    "Date": "date",
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Adj Close": "adj_close",
                    "Volume": "volume",
                },
                inplace=True,
            )
            # Handle missing 'Adj Close' data
            if df["adj_close"].isnull().any():
                logger.warning(
                    f"'Adj Close' data is missing for ETF {symbol}. Using 'Close' prices instead."
                )
                df["adj_close"].fillna(df["close"], inplace=True)
            # Use adjusted close
            df["close"] = df["adj_close"]
            # Handle any remaining missing 'close' prices
            if df["close"].isnull().any():
                logger.warning(
                    f"'Close' data contains missing values for ETF {symbol}. Forward filling missing values."
                )
                df["close"].fillna(method="ffill", inplace=True)
                df["open"].fillna(method="ffill", inplace=True)
                df["high"].fillna(method="ffill", inplace=True)
                df["low"].fillna(method="ffill", inplace=True)
                df["volume"].fillna(0, inplace=True)  # Set missing volume to 0
            # Set index
            df.set_index("date", inplace=True)
            df.sort_index(inplace=True)

            data_feed = bt.feeds.PandasData(
                dataname=df,
                name=symbol,
                fromdate=start_date,
                todate=end_date,
                open="open",
                high="high",
                low="low",
                close="close",
                volume="volume",
                openinterest=-1,
            )
            data_feeds[symbol] = data_feed
            logger.info(f"Data for ETF {symbol} fetched and prepared.")
        except Exception as e:
            logger.error(f"Failed to fetch data for ETF {symbol}: {e}")
            raise
    return data_feeds


def prepare_economic_data(
    indicators: Dict[str, Any], logger: logging.Logger
) -> Dict[str, List[Dict[Hashable, Any]]]:
    """
    Prepare economic data for the strategy.

    Args:
        indicators (Dict[str, Any]): The indicators data fetched.
        logger (logging.Logger): Logger instance.

    Returns:
        Dict[str, List[Dict[Hashable, Any]]]: Prepared economic data.
    """
    economic_data = {}
    for indicator in indicators.values():
        if isinstance(indicator.data, pd.DataFrame):
            records_df = indicator.data.reset_index()
            records_df["date"] = records_df["date"].dt.strftime("%Y-%m-%d")
            economic_data[indicator.internal_key] = records_df.to_dict(orient="records")
            logger.debug(
                f"Converted DataFrame for '{indicator.internal_key}' to list of records with 'date' as string."
            )
        else:
            # Assume it's already a list of dicts and validate
            validated_entries = []
            for entry in indicator.data:
                if "date" not in entry:
                    logger.error(
                        f"Missing 'date' in data for indicator '{indicator.internal_key}': {entry}"
                    )
                    raise KeyError(
                        f"Missing 'date' in data for indicator '{indicator.internal_key}'"
                    )
                # Convert 'date' to string if it's a datetime or Timestamp
                if isinstance(entry["date"], (datetime, pd.Timestamp)):
                    entry["date"] = entry["date"].strftime("%Y-%m-%d")
                elif isinstance(entry["date"], str):
                    # Validate date format
                    try:
                        datetime.strptime(entry["date"], "%Y-%m-%d")
                    except ValueError as ve:
                        logger.error(
                            f"Invalid date format in data for indicator '{indicator.internal_key}': {entry} - {ve}"
                        )
                        raise
                else:
                    logger.error(
                        f"Unexpected type for 'date' in data for indicator '{indicator.internal_key}': {entry}"
                    )
                    raise TypeError(
                        f"Unexpected type for 'date' in data for indicator '{indicator.internal_key}'"
                    )
                validated_entries.append(entry)
            economic_data[indicator.internal_key] = validated_entries
            logger.debug(
                f"Validated and formatted list of dicts for '{indicator.internal_key}'."
            )
    return economic_data


def extract_performance_metrics(
    strategy: bt.Strategy, cerebro: bt.Cerebro, initial_capital: float
) -> Dict[str, Any]:
    """
    Extract performance metrics from the strategy and cerebro.

    Args:
        strategy (bt.Strategy): The strategy instance after backtest.
        cerebro (bt.Cerebro): The cerebro instance used for backtesting.
        initial_capital (float): The initial capital used in the backtest.

    Returns:
        Dict[str, Any]: A dictionary containing performance metrics.
    """
    final_value = cerebro.broker.getvalue()
    sharpe = strategy.analyzers.sharpe.get_analysis()
    drawdown = strategy.analyzers.drawdown.get_analysis()

    # Extract max_drawdown
    max_drawdown = drawdown.drawdown

    # Compute cumulative return
    cumulative_return = (final_value / initial_capital) - 1.0

    performance = {
        "final_value": final_value,
        "sharpe_ratio": sharpe.get("sharperatio", "N/A"),
        "max_drawdown": max_drawdown,
        "cumulative_return": cumulative_return,
    }

    return performance


class PortfolioValueObserver(bt.Observer):
    """
    Observer to track and log the portfolio value over time, including the date.
    """

    lines = ("value",)
    plotinfo = dict(plot=True, subplot=True)

    def next(self) -> None:
        # Retrieve the current date from the data feed
        current_date = self._owner.datas[0].datetime.date(0)
        portfolio_value = self._owner.broker.getvalue()
        logger.debug(f"Date: {current_date} - Portfolio Value: ${portfolio_value:.2f}")


class BaseStrategy(bt.Strategy):
    """
    Base strategy class to encapsulate common functionality.
    """

    def __init__(self) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)

    def notify_order(self, order: bt.Order) -> None:
        """
        Called when an order's status changes.

        Args:
            order (bt.Order): The order object.
        """
        if order.status in [order.Submitted, order.Accepted]:
            return  # Order has been submitted/accepted but not yet completed

        if order.status == order.Completed:
            if order.isbuy():
                self.logger.info(
                    f"BUY EXECUTED: {order.executed.size} shares of {order.data._name} at ${order.executed.price:.2f}, "
                    f"Value: ${order.executed.value:.2f}, Comm: ${order.executed.comm:.2f}"
                )
            elif order.issell():
                self.logger.info(
                    f"SELL EXECUTED: {order.executed.size} shares of {order.data._name} at ${order.executed.price:.2f}, "
                    f"Value: ${order.executed.value:.2f}, Comm: ${order.executed.comm:.2f}"
                )
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            price_str = f"${order.price:.2f}" if order.price is not None else "N/A"
            self.logger.warning(
                f"Order {order.getstatusname()} for {order.data._name}: "
                f"Size: {order.size}, Price: {price_str}"
            )
        else:
            self.logger.debug(
                f"Order status changed: {order.getstatusname()} for {order.data._name}"
            )

    def notify_trade(self, trade: bt.Trade) -> None:
        """
        Called when a trade is updated.

        Args:
            trade (bt.Trade): The trade object.
        """
        if trade.isclosed:
            self.logger.info(
                f"TRADE PROFIT, GROSS ${trade.pnl:.2f}, NET ${trade.pnlcomm:.2f}"
            )


# Define custom parameter protocols
class BondAllocationParams(Protocol):
    allocations: Dict[str, float]
    rebalance_frequency: int
    economic_data: Dict[str, Any]
    symbol_mapping: Dict[str, str]
    allocation_constraints: Dict[str, Any]


class BuyAndHoldParams(Protocol):
    etf_symbol: str


# Strategy classes
class BondAllocationStrategy(BaseStrategy):
    """
    Backtrader Strategy for Bond Portfolio Allocation.
    """

    params = (  # type: ignore
        ("allocations", {"VBIRX": 40.0, "VBLAX": 40.0, "VTAPX": 20.0}),
        ("rebalance_frequency", 12),  # Annual rebalance
        ("economic_data", {}),
        ("symbol_mapping", VANGUARD_MAPPING),
        ("allocation_constraints", {}),
    )

    def __init__(self) -> None:
        super().__init__()
        self.logger.debug("Initializing BondAllocationStrategy.")

        # Cast self.params to BondAllocationParams and assign to an instance variable
        self.params_casted: BondAllocationParams = cast(
            BondAllocationParams, self.params
        )

        # Initialize Portfolio with allocations and constraints
        allocations = self.params_casted.allocations
        allocation_constraints = self.params_casted.allocation_constraints
        self.portfolio = Portfolio(
            allocations=allocations,
            min_allocations={
                asset: allocation_constraints[asset]["min"] for asset in allocations
            },
            max_allocations={
                asset: allocation_constraints[asset]["max"] for asset in allocations
            },
            logger=self.logger,
        )
        self.logger.debug(f"Initial allocations: {self.portfolio.get_allocations()}")

        # Initialize DecisionEngine
        self.decision_engine = DecisionEngine(
            indicators={},  # To be updated during rebalance
            portfolio=self.portfolio,
            allocation_constraints=allocation_constraints,
            logger=self.logger,
        )

        # Economic data and symbol mapping
        self.economic_data = self.params_casted.economic_data
        self.symbol_mapping = self.params_casted.symbol_mapping

        # Track the last rebalance year to ensure annual rebalancing
        self.last_rebalance_year: int = 0

    def next(self) -> None:
        """
        Called on each new bar. Checks if it's time to rebalance and performs rebalancing.
        """
        current_date: datetime = self.datas[0].datetime.date(0)
        current_year: int = current_date.year
        current_month: int = current_date.month

        # Perform initial rebalance if not done yet
        if self.last_rebalance_year == 0:
            self.logger.info(f"Performing initial rebalance on {current_date}")
            self.perform_rebalance(current_date)
            self.last_rebalance_year = current_year
        elif current_month == 12 and current_year != self.last_rebalance_year:
            self.logger.info(f"Rebalancing on {current_date}")
            self.perform_rebalance(current_date)
            self.last_rebalance_year = current_year

    def perform_rebalance(self, rebalance_date: datetime) -> None:
        """
        Performs the portfolio rebalancing based on economic indicators up to the rebalance_date.

        Args:
            rebalance_date (datetime): The date to perform rebalancing.
        """
        self.logger.debug(f"Starting rebalance for date: {rebalance_date}")

        # Collect economic indicators up to rebalance_date
        indicators: Dict[str, EconomicIndicator] = {}
        for key, data in self.economic_data.items():
            filtered_data = [
                entry
                for entry in data
                if datetime.strptime(entry["date"], "%Y-%m-%d").date() <= rebalance_date
            ]
            indicator_config = next(
                (ind for ind in INDICATORS if ind.internal_key == key), None
            )
            if not indicator_config:
                self.logger.error(f"No IndicatorConfig found for key: {key}")
                continue  # Skip this indicator

            indicators[key] = EconomicIndicator(
                config=indicator_config,
                data=filtered_data,
                rebalancing_date=rebalance_date.strftime("%Y-%m-%d"),
                logger=self.logger.getChild(key),
            )
            self.logger.info(f"EconomicIndicator for '{key}' created successfully.")

        self.logger.debug(
            f"Collected {len(indicators)} economic indicators up to {rebalance_date}"
        )

        # Re-initialize DecisionEngine with new indicators
        self.decision_engine = DecisionEngine(
            indicators=indicators,
            portfolio=self.portfolio,
            allocation_constraints=self.params_casted.allocation_constraints,
            logger=self.logger,
        )
        self.decision_engine.apply_decision_rules()

        self.logger.debug(
            f"Allocations after decision rules: {self.portfolio.get_allocations()}"
        )

        # Rebalance the portfolio based on the new allocations
        self.rebalance_portfolio()

    def rebalance_portfolio(self) -> None:
        """
        Rebalances the portfolio based on the current allocations using Backtrader's order_target_size.
        """
        total_value = self.broker.getvalue()
        for internal_key, percentage in self.portfolio.get_allocations().items():
            etf_symbol = self.symbol_mapping.get(internal_key)
            if not etf_symbol:
                continue

            data = self.getdatabyname(etf_symbol)
            price = data.close[0]
            if pd.isna(price):
                self.logger.warning(
                    f"Price data is missing for {etf_symbol} on {self.datas[0].datetime.date(0)}; skipping order."
                )
                continue  # Skip ordering this ETF

            target_value = total_value * (percentage / 100.0)
            current_position = self.getposition(data).size * price
            order_value = target_value - current_position

            # Calculate the number of shares to order
            size = int(order_value / price)

            # Ensure we don't order on margin
            if size > 0 and (size * price) > self.broker.getcash():
                self.logger.warning(
                    f"Not enough cash to buy {size} shares of {etf_symbol}; adjusting order size."
                )
                size = int(self.broker.getcash() / price)

            # Place the order if size is non-zero
            if size != 0:
                self.order_target_size(data=data, target=size)


class BuyAndHoldBNDStrategy(BaseStrategy):
    """
    Backtrader Strategy for a Buy-and-Hold approach on BND.
    """

    params = (("etf_symbol", "BND"),)

    def __init__(self) -> None:
        super().__init__()
        self.logger.debug("Initializing BuyAndHoldBNDStrategy.")
        self.order = None
        self.bought = False

        # Cast self.params to BuyAndHoldParams and assign to an instance variable
        self.params_casted: BuyAndHoldParams = cast(BuyAndHoldParams, self.params)

    def next(self) -> None:
        """
        Called on each new bar. Buys BND on the first available bar with valid price data and holds.
        """
        if not self.bought and self.order is None:
            etf_symbol = self.params_casted.etf_symbol
            price = self.getdatabyname(etf_symbol).close[0]
            if pd.isna(price):
                self.logger.warning(
                    f"Price data is missing on {self.datas[0].datetime.date(0)}; cannot place order."
                )
                return  # Skip this bar and try again on the next bar
            else:
                self.logger.info(
                    f"Placing initial buy order for {etf_symbol} at price ${price:.2f}."
                )
                self.order = self.buy(data=self.getdatabyname(etf_symbol))

    def notify_order(self, order: bt.Order) -> None:
        """
        Called when an order's status changes.
        """
        super().notify_order(order)  # Maintain existing logging

        if order.status == order.Completed:
            if order.isbuy():
                self.bought = True
                self.logger.info(
                    f"BUY EXECUTED: {order.executed.size} shares of {order.data._name} at ${order.executed.price:.2f}, "
                    f"Value: ${order.executed.value:.2f}, Comm: ${order.executed.comm:.2f}"
                )
            elif order.issell():
                self.logger.info(
                    f"SELL EXECUTED: {order.executed.size} shares of {order.data._name} at ${order.executed.price:.2f}, "
                    f"Value: ${order.executed.value:.2f}, Comm: ${order.executed.comm:.2f}"
                )
            self.order = None  # Reset the order
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.logger.warning(
                f"Order {order.getstatusname()} for {order.data._name}: "
                f"Size: {order.size}, Price: ${order.price:.2f}"
                if order.price
                else "N/A"
            )
            self.order = None  # Reset order to allow retrying if desired
        else:
            # Handle other statuses if necessary
            self.logger.debug(
                f"Order status {order.getstatusname()} for {order.data._name}"
            )

    def notify_trade(self, trade: bt.Trade) -> None:
        """
        Called when a trade is updated.
        """
        super().notify_trade(trade)  # Maintain existing logging

        if trade.isclosed:
            self.logger.info(
                f"TRADE PROFIT, GROSS ${trade.pnl:.2f}, NET ${trade.pnlcomm:.2f}"
            )


def run_backtest(
    strategy_class: Any,
    strategy_params: Dict[str, Any],
    data_feeds: Dict[str, bt.feeds.PandasData],
    initial_capital: float,
    commission: float,
) -> Tuple[Any, Dict[str, Any]]:
    """
    Runs a single backtest with the specified strategy and parameters.

    Args:
        strategy_class (Any): The strategy class to run.
        strategy_params (Dict[str, Any]): Parameters to pass to the strategy.
        data_feeds (Dict[str, bt.feeds.PandasData]): Dictionary of data feeds.
        initial_capital (float): Starting capital for the backtest.
        commission (float): Commission rate per trade.

    Returns:
        Tuple[Any, Dict[str, Any]]: The strategy instance and its performance metrics.
    """
    cerebro = bt.Cerebro()

    # Add the strategy with parameters
    cerebro.addstrategy(strategy_class, **strategy_params)

    # Add data feeds
    for etf, data_feed in data_feeds.items():
        cerebro.adddata(data_feed, name=etf)
        logger.debug(f"Added data feed for ETF {etf}.")

    # Set initial capital and commission
    cerebro.broker.setcash(initial_capital)
    cerebro.broker.setcommission(commission=commission)

    # Add observers and analyzers
    cerebro.addobserver(PortfolioValueObserver)
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe")
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name="timereturn")

    # Run the backtest
    logger.debug(f"Running backtest for strategy {strategy_class.__name__}.")
    results = cerebro.run()
    strategy = results[0]

    # Extract performance metrics
    performance = extract_performance_metrics(strategy, cerebro, initial_capital)

    logger.debug(
        f"Backtest for {strategy_class.__name__} completed. Performance: {performance}"
    )

    return strategy, performance


def main() -> None:
    """
    Main function to set up and run the backtests for multiple strategies.
    """
    logger.info("Starting backtests for multiple strategies.")

    # Define configuration file path
    config_path: str = "config/config.yaml"

    # Define required configuration fields
    required_config_fields: List[str] = [
        "storage.fred_file_path",
        "storage.recommendation_file_path",
        "api.fred_api_key",
        "allocations",
    ]

    # Load configuration
    config: Dict[str, Any] = load_configuration(
        config_path, logger, required_config_fields
    )

    # Parse allocations to extract allocation_constraints
    (
        initial_allocations,
        min_allocations,
        max_allocations,
        allocation_constraints,
    ) = parse_allocations(config, logger)
    logger.debug(f"Initial Allocations: {initial_allocations}")
    logger.debug(f"Min Allocations: {min_allocations}")
    logger.debug(f"Max Allocations: {max_allocations}")
    logger.debug(f"Allocation Constraints: {allocation_constraints}")

    # Initialize data storage and fetcher
    fetcher = initialize_data_storage_and_fetcher(config, logger)

    # Fetch economic indicators
    indicators = collect_economic_indicators(
        fetcher, config.get("indicators", INDICATORS), logger
    )

    # Prepare economic_data dictionary for the strategy
    economic_data = prepare_economic_data(indicators, logger)

    # Fetch historical ETF data (2013-2023)
    etf_start_date: datetime = datetime(2013, 1, 1)
    etf_end_date: datetime = datetime(2023, 12, 31)
    etf_symbols = list(VANGUARD_MAPPING.values())
    bond_data_feeds = prepare_etf_data(etf_symbols, etf_start_date, etf_end_date)

    # Define the strategies you want to test
    strategies: List[Tuple[str, Any]] = [
        ("BondAllocationStrategy", BondAllocationStrategy),
        ("BuyAndHoldBNDStrategy", BuyAndHoldBNDStrategy),
    ]

    # List to store the results
    final_results: List[Dict[str, Any]] = []

    # Dictionary to store equity curves for optional plotting
    equity_curves: Dict[str, pd.Series] = {}

    # Set initial capital and commission
    initial_capital: float = 100000.0
    commission: float = 0.001  # 0.1%

    for strat_name, strat_class in strategies:
        logger.info(f"Running backtest for {strat_name}.")

        # Define strategy-specific parameters
        if strat_name == "BondAllocationStrategy":
            strategy_params = {
                "economic_data": economic_data,
                "symbol_mapping": VANGUARD_MAPPING,
                "allocations": initial_allocations,
                "allocation_constraints": allocation_constraints,
            }
        elif strat_name == "BuyAndHoldBNDStrategy":
            strategy_params = {
                "etf_symbol": "BND",
            }
        else:
            strategy_params = {}

        # Run the backtest
        try:
            strategy_instance, performance = run_backtest(
                strategy_class=strat_class,
                strategy_params=strategy_params,
                data_feeds=bond_data_feeds,
                initial_capital=initial_capital,
                commission=commission,
            )
        except Exception as e:
            logger.critical(f"Backtest for {strat_name} failed: {e}")
            continue

        # Store the performance metrics
        final_results.append(
            {
                "strategy": strat_name,
                "final_value": performance["final_value"],
                "sharpe_ratio": performance["sharpe_ratio"],
                "max_drawdown": performance["max_drawdown"],
                "cumulative_return": performance["cumulative_return"],
            }
        )

        # Collect equity curve for plotting
        timereturn = strategy_instance.analyzers.timereturn.get_analysis()
        equity_curve = pd.Series(timereturn).add(1.0).cumprod().mul(initial_capital)
        equity_curves[strat_name] = equity_curve

    # Compare the results
    logger.info("Backtest Results:")
    for res in final_results:
        logger.info(f"Strategy: {res['strategy']}")
        logger.info(f"Final Portfolio Value: ${res['final_value']:,.2f}")
        logger.info(f"Sharpe Ratio: {res['sharpe_ratio']}")
        if isinstance(res["max_drawdown"], float):
            logger.info(f"Max Drawdown: {res['max_drawdown']:.2f}%")
        else:
            logger.info(f"Max Drawdown: {res['max_drawdown']}%")
        logger.info(f"Cumulative Return: {res['cumulative_return'] * 100:.2f}%")
        logger.info("----------------------------")

    # Optional: Plot the equity curves for visual comparison
    try:
        plt.figure(figsize=(12, 6))
        for strat_name, equity in equity_curves.items():
            # Ensure index is a DatetimeIndex
            if not isinstance(equity.index, pd.DatetimeIndex):
                equity.index = pd.to_datetime(equity.index)
            # Plot the equity curve
            plt.plot(equity.index, equity.values.astype(float), label=strat_name)
        plt.legend()
        plt.title("Equity Curve Comparison")
        plt.xlabel("Date")
        plt.ylabel("Portfolio Value ($)")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    except Exception as e:
        logger.error(f"Failed to plot equity curves: {e}")

    logger.info("Backtests completed.")


if __name__ == "__main__":
    main()
