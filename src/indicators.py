# src/indicators.py

"""
indicators.py

Module for Economic Indicator analysis.

This module contains the `EconomicIndicator` class, responsible for processing,
analyzing, and generating insights from economic data based on specified configurations.
It handles the loading of data, calculation of statistics, and generation of trend signals.
"""

import logging
import math
from typing import Any, Dict, List, Optional

import pandas as pd

from .config import IndicatorConfig


class EconomicIndicator:
    """
    Processes and analyzes economic data to generate insights and trend signals.

    The `EconomicIndicator` class handles the ingestion of raw economic data, processes it
    into a structured format, calculates relevant statistics, and generates trend signals
    based on specified configurations. It supports various calculation methods such as
    percentage change, basis points change, z-score, year-over-year change, and CAGR.

    Attributes:
        config (IndicatorConfig): Configuration details for the economic indicator.
        data (pd.DataFrame): Cleaned and processed data for analysis.
        time_frame_weights (Dict[str, float]): Weights assigned to different time frames for weighted analysis.
        indicator_type (str): Frequency type of the indicator (e.g., DAILY, MONTHLY, QUARTERLY).
        name (str): Name of the economic indicator.
        internal_key (str): Internal identifier for the indicator.
        logger (logging.Logger): Logger instance for logging activities and debugging.
    """

    def __init__(
        self,
        config: IndicatorConfig,
        data: List[Dict[str, Any]],
        logger: Optional[logging.Logger] = None,
    ) -> None:
        """
        Initialize the EconomicIndicator with configuration and data.

        Args:
            config (IndicatorConfig): Configuration instance containing details like calculation method,
                                      thresholds, and time frame weights.
            data (List[Dict[str, Any]]): Raw data points for the indicator, typically fetched from an external source.
            logger (Optional[logging.Logger], optional): Logger instance for logging messages.
                                                         If None, a default logger named "Bondit.EconomicIndicator" is used.

        Example:
            >>> config = IndicatorConfig(
            ...     name="CPI",
            ...     internal_key="cpi",
            ...     indicator_type="MONTHLY",
            ...     calculation_method="percentage_change",
            ...     thresholds=(2.0, -2.0),
            ...     time_frame_weights={"1y": 0.5, "3y": 0.3, "5y": 0.2}
            ... )
            >>> data = [
            ...     {"date": "2023-09-01", "value": 250.5},
            ...     {"date": "2023-08-01", "value": 249.1},
            ...     # More data points...
            ... ]
            >>> cpi_indicator = EconomicIndicator(config, data)
        """
        self.logger = logger or logging.getLogger("Bondit.EconomicIndicator")
        self.config = config
        self.time_frame_weights = config.time_frame_weights
        self.indicator_type: str = config.indicator_type
        self.name = config.name
        self.internal_key = config.internal_key
        self.data: pd.DataFrame = pd.DataFrame()
        self.logger.debug(f"Initializing EconomicIndicator for '{self.name}'.")
        self.process_data(data)

    def process_data(self, data_points: List[Dict[str, Any]]) -> None:
        """
        Process raw data points into a cleaned DataFrame.

        Converts raw data into a Pandas DataFrame, ensuring proper date and value formatting.
        This method also sorts the data in reverse chronological order to facilitate analysis.

        Args:
            data_points (List[Dict[str, Any]]): Raw data points for the indicator.

        Example:
            >>> data_points = [
            ...     {"date": "2023-09-01", "value": 250.5},
            ...     {"date": "2023-08-01", "value": 249.1},
            ... ]
            >>> cpi_indicator.process_data(data_points)
        """
        if not data_points:
            self.logger.warning(f"No data provided for indicator '{self.name}'.")
            return

        original_df = pd.DataFrame(data_points)
        if "date" not in original_df.columns or "value" not in original_df.columns:
            self.logger.warning(
                f"Data for '{self.name}' is missing 'date' or 'value' columns."
            )
            return

        self.logger.debug(f"Processing data for '{self.name}'.")

        # Convert 'date' to datetime and 'value' to numeric, coercing errors to NaN
        original_df["date"] = pd.to_datetime(original_df["date"], errors="coerce")
        original_df["value"] = pd.to_numeric(original_df["value"], errors="coerce")

        # Keep the NaN values for alignment
        cleaned_df = original_df.copy()
        cleaned_df.set_index("date", inplace=True)
        cleaned_df.sort_index(
            ascending=False, inplace=True
        )  # Reverse chronological order

        self.data = cleaned_df
        self.logger.info(
            f"Processed data for '{self.name}' with {len(cleaned_df)} records, including NaN values for alignment."
        )

    def extract_time_frame_data(self, years: int) -> Optional[pd.DataFrame]:
        """
        Extract data for a specific time frame.

        Filters the data to include only records within the specified number of years from the latest date.

        Args:
            years (int): Number of years for the time frame.

        Returns:
            Optional[pd.DataFrame]: DataFrame containing data for the specified time frame,
                                    or None if insufficient data is available.

        Example:
            >>> cpi_data = cpi_indicator.extract_time_frame_data(3)
        """
        if self.data.empty:
            self.logger.warning(f"No data available for indicator '{self.name}'.")
            return None

        end_date: pd.Timestamp = self.data.index.max()
        start_date: pd.Timestamp = end_date - pd.DateOffset(years=years)
        time_frame_data: pd.DataFrame = self.data[
            (self.data.index >= start_date) & (self.data.index <= end_date)
        ]

        if time_frame_data.empty:
            self.logger.warning(
                f"No data available for {years}-year period for indicator '{self.name}'."
            )
            return None

        self.logger.debug(
            f"Extracted {len(time_frame_data)} records for {years}-year period for '{self.name}'."
        )
        return time_frame_data

    def calculate_statistics(self, data: pd.Series) -> Dict[str, Any]:
        """
        Calculate basic statistics for a given data series.

        Computes metrics such as the most recent value, average, median, minimum, maximum,
        standard deviation, and z-score to understand the data distribution.

        Args:
            data (pd.Series): Series of numerical values representing the economic indicator.

        Returns:
            Dict[str, Any]: Dictionary containing calculated statistics.

        Example:
            >>> stats = cpi_indicator.calculate_statistics(cpi_data["value"])
        """
        self.logger.debug(f"Calculating statistics for '{self.name}'.")

        # Drop NaN values to ensure accurate calculations
        data = data.dropna()

        if data.empty:
            self.logger.warning(
                f"No valid data to calculate statistics for '{self.name}'."
            )
            return {
                "most_recent_value": None,
                "average": None,
                "median": None,
                "min": None,
                "max": None,
                "std_dev": None,
                "z_score": None,
            }

        stats = {
            "most_recent_value": data.iloc[0],
            "average": data.mean(),
            "median": data.median(),
            "min": data.min(),
            "max": data.max(),
            "std_dev": data.std(),
            "z_score": self.calculate_z_score(data),
        }
        self.logger.debug(f"Statistics for '{self.name}': {stats}")
        return stats

    def calculate_moving_average(self, data: pd.Series, window: int) -> Optional[float]:
        """
        Calculate the moving average over a specified window.

        Computes the average of the most recent 'window' number of data points.

        Args:
            data (pd.Series): Series of numerical values.
            window (int): Number of periods to include in the moving average.

        Returns:
            Optional[float]: Calculated moving average, or None if insufficient data.

        Example:
            >>> moving_avg = cpi_indicator.calculate_moving_average(cpi_data["value"], 12)
        """
        # Drop NaN values to ensure accurate calculations
        data = data.dropna()

        if len(data) < window:
            self.logger.debug(
                f"Insufficient data to calculate moving average for '{self.name}'. Required: {window}, Available: {len(data)}"
            )
            return None

        moving_avg = float(data.iloc[:window].mean())
        self.logger.debug(
            f"Calculated moving average for '{self.name}' with window {window}: {moving_avg}"
        )
        return moving_avg

    def calculate_change(
        self,
        current_value: float,
        moving_average: Optional[float],
        data_series: pd.Series,
        periods_per_year: int,
    ) -> Optional[float]:
        """
        Calculate the change based on the specified calculation method.

        Supports various methods such as percentage change, basis points change, z-score,
        absolute change, year-over-year change, CAGR, and using the current value directly.

        Args:
            current_value (float): The most recent value of the indicator.
            moving_average (Optional[float]): The moving average value for comparison.
            data_series (pd.Series): Series of historical data points.
            periods_per_year (int): Number of periods per year based on indicator frequency.

        Returns:
            Optional[float]: Calculated change value, or None if calculation is not possible.

        Example:
            >>> change = cpi_indicator.calculate_change(250.5, 245.0, cpi_data["value"], 12)
        """
        method = self.config.calculation_method
        self.logger.debug(
            f"Calculating change for '{self.name}' using method '{method}'."
        )

        # Ensure required values are available for certain methods
        if method in ["percentage_change", "basis_points_change", "absolute_change"]:
            if moving_average is None or pd.isna(moving_average):
                self.logger.warning(
                    f"Cannot calculate {method} for '{self.name}': moving_average is None or NaN."
                )
                return None

        if method == "percentage_change":
            if moving_average == 0.0:
                self.logger.warning(
                    f"Cannot calculate percentage change for '{self.name}': moving_average is zero."
                )
                return None

            assert (
                moving_average is not None
            )  # Inform MyPy that moving_average is a float here
            change = ((current_value - moving_average) / moving_average) * 100.0
            self.logger.debug(f"Percentage change for '{self.name}': {change:.2f}%.")
            return change

        elif method == "basis_points_change":
            if moving_average is None:
                self.logger.warning(
                    f"Cannot calculate basis points change for '{self.name}': moving_average is None."
                )
                return None
            assert moving_average is not None  # Inform MyPy
            change = (current_value - moving_average) * 100  # Basis points
            self.logger.debug(
                f"Basis points change for '{self.name}': {change:.2f} bps."
            )
            return change

        elif method == "z_score":
            z_score = self.calculate_z_score(data_series)
            self.logger.debug(f"Z-score for '{self.name}': {z_score}")
            return z_score

        elif method == "absolute_change":
            if moving_average is None:
                self.logger.warning(
                    f"Cannot calculate absolute change for '{self.name}': moving_average is None."
                )
                return None
            assert (
                moving_average is not None
            )  # Inform MyPy that moving_average is a float here
            change = current_value - moving_average
            self.logger.debug(f"Absolute change for '{self.name}': {change}")
            return change

        elif method == "year_over_year_change":
            yoy_change = self.calculate_year_over_year_change(data_series)
            self.logger.debug(f"Year-over-year change for '{self.name}': {yoy_change}")
            return yoy_change

        elif method == "cagr":
            cagr = self.calculate_cagr(data_series, periods_per_year)
            self.logger.debug(f"CAGR for '{self.name}': {cagr}")
            return cagr

        elif method == "current_value":
            self.logger.debug(f"Current value for '{self.name}': {current_value}")
            return current_value

        else:
            self.logger.warning(
                f"Unknown calculation method '{method}' for '{self.name}'."
            )
            return None

    def calculate_z_score(self, series: pd.Series) -> Optional[float]:
        """
        Calculate the z-score for the most recent value in the series.

        The z-score measures how many standard deviations the most recent value is from the median.
        It is scaled using MAD (Median Absolute Deviation) to ensure robustness against outliers.

        Args:
            series (pd.Series): Series of numerical values in reverse chronological order.

        Returns:
            Optional[float]: Calculated z-score, or None if calculation is not possible.

        Example:
            >>> z = cpi_indicator.calculate_z_score(cpi_data["value"])
        """
        series = series.dropna()
        if len(series) <= 1:
            self.logger.debug(
                f"Insufficient data to calculate z-score for '{self.name}'. Required: >1, Available: {len(series)}"
            )
            return None

        median: float = float(series.median())
        mad: float = float((series - median).abs().median())

        if mad == 0.0:
            self.logger.debug(
                f"MAD is zero for '{self.name}', cannot calculate z-score."
            )
            return None  # Avoid division by zero

        # Scaling MAD to estimate standard deviation
        scaled_mad: float = mad * 1.4826
        value: float = float(series.iloc[0])

        z_score: float = (value - median) / scaled_mad
        self.logger.debug(f"Z-score for '{self.name}': {z_score:.2f}")
        return z_score

    def calculate_year_over_year_change(
        self, data_series: pd.Series
    ) -> Optional[float]:
        """
        Calculate the year-over-year (YoY) change for the indicator.

        Compares the most recent value with the value from one year ago to determine growth or decline.

        Args:
            data_series (pd.Series): Series of numerical values indexed by date.

        Returns:
            Optional[float]: YoY change as a percentage, or None if calculation is not possible.

        Example:
            >>> yoy = cpi_indicator.calculate_year_over_year_change(cpi_data["value"])
        """
        # Drop NaN values to ensure accurate calculations
        data_series = data_series.dropna()

        if data_series.empty:
            self.logger.warning(
                f"No valid data for year-over-year calculation for '{self.name}'."
            )
            return None

        current_date = data_series.index[0]
        one_year_ago_date = current_date - pd.DateOffset(years=1)

        # Ensure the index is a DatetimeIndex
        if not isinstance(data_series.index, pd.DatetimeIndex):
            self.logger.error(
                f"Index is not a DatetimeIndex. It is {type(data_series.index)}."
            )
            return None

        # Create a sorted copy in ascending order for accurate date matching
        sorted_series = data_series.sort_index(ascending=True)

        # Find the closest available date using get_indexer
        indexer = sorted_series.index.get_indexer([one_year_ago_date], method="nearest")
        pos = indexer[0]

        if pos == -1:
            self.logger.warning(f"Cannot find data one year ago for '{self.name}'.")
            return None

        value_one_year_ago_raw = sorted_series.iloc[pos]

        # Ensure the raw value is a float
        try:
            value_one_year_ago: float = float(value_one_year_ago_raw)
            current_value: float = float(data_series.iloc[0])  # Original series
        except (ValueError, TypeError) as e:
            self.logger.error(f"Error casting values to float: {e}")
            return None

        if pd.isna(current_value) or pd.isna(value_one_year_ago):
            self.logger.warning(
                f"Cannot calculate year-over-year change for '{self.name}': current or year-ago value is NaN."
            )
            return None

        assert current_value is not None, "current_value should not be None here."
        assert (
            value_one_year_ago is not None
        ), "value_one_year_ago should not be None here."

        if value_one_year_ago == 0.0:
            self.logger.warning(
                f"Cannot calculate year-over-year change for '{self.name}': value one year ago is zero."
            )
            return None

        yoy_change = ((current_value - value_one_year_ago) / value_one_year_ago) * 100.0
        self.logger.debug(
            f"Year-over-year change for '{self.name}': {yoy_change:.2f}%."
        )
        return yoy_change

    def calculate_cagr(self, data: pd.Series, periods_per_year: int) -> Optional[float]:
        """
        Calculate the Compound Annual Growth Rate (CAGR) for the indicator.

        CAGR represents the mean annual growth rate of an investment over a specified time period longer than one year.

        Args:
            data (pd.Series): Series of numerical values indexed by date.
            periods_per_year (int): Number of periods per year based on indicator frequency.

        Returns:
            Optional[float]: CAGR as a percentage, or None if calculation is not possible.

        Example:
            >>> cagr = cpi_indicator.calculate_cagr(cpi_data["value"], 12)
        """
        # Drop NaN values to ensure accurate calculations
        data = data.dropna()

        if data.empty:
            self.logger.warning(f"No valid data to calculate CAGR for '{self.name}'.")
            return None

        start_date = data.index[-1]
        end_date = data.index[0]
        n_years = (end_date - start_date).days / 365.25

        if n_years <= 0:
            self.logger.warning(
                f"Invalid time span to calculate CAGR for '{self.name}'."
            )
            return None

        start_value: float = float(data.iloc[-1])
        end_value: float = float(data.iloc[0])

        if start_value <= 0.0 or end_value <= 0.0:
            self.logger.warning(
                f"Cannot calculate CAGR for '{self.name}': start or end value is non-positive."
            )
            return None

        try:
            cagr: float = (end_value / start_value) ** (1.0 / n_years) - 1.0
            cagr_percentage = cagr * 100.0
            self.logger.debug(f"CAGR for '{self.name}': {cagr_percentage:.2f}%")
            return cagr_percentage
        except (ZeroDivisionError, OverflowError) as e:
            self.logger.error(f"CAGR calculation error for '{self.name}': {e}")
            return None

    def generate_trend_signal(self, change: Optional[float]) -> str:
        """
        Generate a trend signal based on the calculated change and predefined thresholds.

        Determines whether the trend is 'Rising', 'Falling', or 'Stable' based on the change
        relative to upper and lower thresholds defined in the configuration.

        Args:
            change (Optional[float]): The calculated change value from the specified method.

        Returns:
            str: Trend signal indicating 'Rising', 'Falling', or 'Stable'.

        Example:
            >>> signal = cpi_indicator.generate_trend_signal(1.5)
            >>> print(signal)
            Stable
        """
        if change is None or math.isnan(change):
            self.logger.debug(
                f"No change data to generate trend signal for '{self.name}'."
            )
            return "No Data"

        assert change is not None  # Inform MyPy that change is a float
        upper_threshold, lower_threshold = self.config.thresholds
        self.logger.debug(
            f"Generating trend signal for '{self.name}' with change: {change} "
            f"and thresholds: upper={upper_threshold}, lower={lower_threshold}."
        )

        if self.config.calculation_method in [
            "year_over_year_change",
            "cagr",
            "current_value",
        ]:
            # Both thresholds are positive numbers for these methods
            if change > upper_threshold:
                self.logger.debug(f"Trend signal for '{self.name}': Rising.")
                return "Rising"
            elif change < lower_threshold:
                self.logger.debug(f"Trend signal for '{self.name}': Falling.")
                return "Falling"
            else:
                self.logger.debug(f"Trend signal for '{self.name}': Stable.")
                return "Stable"
        else:
            # Thresholds are symmetric around zero for other methods
            if change > upper_threshold:
                self.logger.debug(f"Trend signal for '{self.name}': Rising.")
                return "Rising"
            elif change < lower_threshold:
                self.logger.debug(f"Trend signal for '{self.name}': Falling.")
                return "Falling"
            else:
                self.logger.debug(f"Trend signal for '{self.name}': Stable.")
                return "Stable"

    def analyze_indicator(self) -> Dict[str, Any]:
        """
        Analyze the economic indicator across different time frames and aggregate results.

        Applies analysis methods based on configuration to determine trends and statistics.
        Aggregates weighted changes and determines the overall trend signal.

        Returns:
            Dict[str, Any]: Dictionary containing analysis results, including trend signals,
                            weighted changes, and statistics.

        Example:
            >>> analysis_results = cpi_indicator.analyze_indicator()
        """
        results: Dict[str, Any] = {}
        weighted_change_sum: float = 0.0
        total_weight: float = 0.0
        trend_directions: List[int] = []  # 1 for Rising, -1 for Falling, 0 for Stable

        if self.data.empty:
            self.logger.warning(f"No data to analyze for '{self.name}'.")
            return {"error": f"No data for indicator '{self.name}'"}

        frequency: str = self.indicator_type
        if frequency == "QUARTERLY":
            periods_per_year = 4
        elif frequency == "MONTHLY":
            periods_per_year = 12
        elif frequency == "DAILY":
            periods_per_year = 252  # Approximate trading days in a year
        else:
            periods_per_year = 1  # Default to annual

        self.logger.debug(
            f"Analyzing '{self.name}' with frequency '{frequency}' and periods_per_year={periods_per_year}."
        )

        for period, weight in self.time_frame_weights.items():
            self.logger.debug(
                f"Analyzing '{self.name}' for {period}-year period with weight {weight}."
            )
            time_frame_data: Optional[pd.DataFrame] = self.extract_time_frame_data(
                period
            )
            if time_frame_data is None or time_frame_data.empty:
                continue

            values: pd.Series = time_frame_data["value"].dropna()
            if values.empty:
                self.logger.warning(
                    f"No valid data points for {period}-year period for '{self.name}'."
                )
                continue

            # Determine window size based on period and frequency
            window = min(len(values), period * periods_per_year)

            # Calculate moving average if the calculation method requires it
            moving_average: Optional[float] = (
                self.calculate_moving_average(values, window)
                if self.config.calculation_method
                in ["percentage_change", "basis_points_change", "absolute_change"]
                else None
            )

            if moving_average is None and self.config.calculation_method in [
                "percentage_change",
                "basis_points_change",
                "absolute_change",
            ]:
                self.logger.warning(
                    f"Moving average could not be calculated for '{self.name}' in {period}-year period."
                )

            current_value: Optional[float] = (
                float(values.iloc[0]) if not values.empty else None
            )

            if current_value is not None:
                # Calculate change based on the specified method
                change = self.calculate_change(
                    current_value, moving_average, values, periods_per_year
                )
            else:
                change = None
                self.logger.warning(
                    f"Current value is None for '{self.name}' in {period}-year period."
                )

            # Generate trend signal using thresholds
            signal: str = self.generate_trend_signal(change)

            # Collect additional metrics if needed
            year_over_year_change = (
                self.calculate_year_over_year_change(values)
                if self.config.calculation_method == "year_over_year_change"
                else None
            )

            cagr = (
                self.calculate_cagr(values, periods_per_year)
                if self.config.calculation_method == "cagr"
                else None
            )

            z_score = (
                self.calculate_z_score(values)
                if self.config.calculation_method == "z_score"
                else None
            )

            results[f"{period}y"] = {
                "moving_average": moving_average,
                "current_value": current_value,
                "change": change,
                "cagr": cagr,
                "signal": signal,
                "z_score": z_score,
                "year_over_year_change": year_over_year_change,
            }

            self.logger.debug(
                f"Analysis for {period}y period of '{self.name}': {results[f'{period}y']}"
            )

            # Update weighted results for overall analysis
            if change is not None and not math.isnan(change):
                weighted_change_sum += change * weight
                total_weight += weight

            # Record trend direction for overall trend determination
            if signal == "Rising":
                trend_directions.append(1)
            elif signal == "Falling":
                trend_directions.append(-1)
            else:
                trend_directions.append(0)

        # Store weighted results
        weighted_change = (
            weighted_change_sum / total_weight if total_weight > 0 else None
        )
        results["weighted"] = {
            "weighted_change": weighted_change,
        }
        self.logger.debug(f"Weighted change for '{self.name}': {weighted_change}")

        # Determine overall trend based on majority of time frames
        rising_count = trend_directions.count(1)
        falling_count = trend_directions.count(-1)
        stable_count = trend_directions.count(0)

        if rising_count > falling_count and rising_count > stable_count:
            overall_trend = "Rising"
        elif falling_count > rising_count and falling_count > stable_count:
            overall_trend = "Falling"
        else:
            overall_trend = "Stable"

        results["overall_trend"] = overall_trend
        self.logger.debug(f"Overall trend for '{self.name}': {overall_trend}")

        # Add statistics to the results
        stats: Dict[str, Any] = self.calculate_statistics(self.data["value"])
        results["statistics"] = stats
        self.logger.debug(f"Statistics for '{self.name}': {stats}")

        self.logger.info(f"Completed analysis for '{self.name}'.")
        return results

    def analyze_all_indicators(self) -> Dict[str, Any]:
        """
        Analyze the economic indicator and return the analysis results.

        This method serves as a wrapper to perform the complete analysis process, encapsulating
        data processing, statistical calculations, and trend signal generation.

        Returns:
            Dict[str, Any]: Comprehensive analysis results for the indicator.

        Example:
            >>> analysis_results = cpi_indicator.analyze_all_indicators()
        """
        self.logger.debug(f"Starting comprehensive analysis for '{self.name}'.")
        analysis = self.analyze_indicator()
        self.logger.debug(
            f"Comprehensive analysis results for '{self.name}': {analysis}"
        )
        return analysis
