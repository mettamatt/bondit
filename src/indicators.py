# src/indicators.py

"""
Module for Economic Indicator Analysis.

This module contains the `EconomicIndicator` class, responsible for processing,
analyzing, and generating insights from economic data based on specified configurations.
It handles the loading of data, calculation of statistics, and generation of trend signals.
"""

import logging
import math
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from .config import FIXED_START_DATE, CalculationMethod, IndicatorConfig, IndicatorType


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
        time_frame_weights (List[TimeFrameWeight]): Weights assigned to different time frames for weighted analysis.
        indicator_type (IndicatorType): Frequency type of the indicator (e.g., DAILY, MONTHLY, QUARTERLY).
        name (str): Name of the economic indicator.
        internal_key (str): Internal identifier for the indicator.
        description (str): Description of the economic indicator.
        logger (logging.Logger): Logger instance for logging activities and debugging.
        rebalancing_date (pd.Timestamp): The date up to which data is analyzed.
    """

    def __init__(
        self,
        config: IndicatorConfig,
        data: List[Dict[str, Any]],
        rebalancing_date: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        """
        Initialize the EconomicIndicator with configuration and data.

        Args:
            config (IndicatorConfig): Configuration instance containing details like calculation method,
                                      thresholds, and time frame weights.
            data (List[Dict[str, Any]]): Raw data points for the indicator, typically fetched from an external source.
            rebalancing_date (Optional[str], optional): The date up to which data should be analyzed in "YYYY-MM-DD" format.
                                                      Defaults to today's date if not provided.
            logger (Optional[logging.Logger], optional): Logger instance for logging messages.
                                                       If None, a default logger named "Bondit.EconomicIndicator" is used.

        Example:
            >>> config = IndicatorConfig(
            ...     name="CPI",
            ...     internal_key="cpi",
            ...     indicator_type=IndicatorType.MONTHLY,
            ...     calculation_method=CalculationMethod.PERCENTAGE_CHANGE,
            ...     thresholds=(2.0, -2.0),
            ...     time_frame_weights=[
            ...         TimeFrameWeight(years=1, weight=0.5),
            ...         TimeFrameWeight(years=3, weight=0.3),
            ...         TimeFrameWeight(years=5, weight=0.2),
            ...     ]
            ... )
            >>> data = [
            ...     {"date": "2023-09-01", "value": 250.5},
            ...     {"date": "2023-08-01", "value": 249.1},
            ...     # More data points...
            ... ]
            >>> cpi_indicator = EconomicIndicator(config, data, rebalancing_date="2023-09-01")
        """
        self.logger: logging.Logger = logger or logging.getLogger(
            "Bondit.EconomicIndicator"
        )
        self.config: IndicatorConfig = config
        self.time_frame_weights = config.time_frame_weights
        self.indicator_type: IndicatorType = config.indicator_type
        self.name: str = config.name
        self.internal_key: str = config.internal_key
        self.description: str = config.description
        self.data: pd.DataFrame = pd.DataFrame()
        self.rebalancing_date: pd.Timestamp = (
            pd.to_datetime(rebalancing_date)
            if rebalancing_date
            else pd.Timestamp.today()
        )

        # Ensure that rebalancing_date is not before FIXED_START_DATE
        FIXED_START_DATE_DT = pd.to_datetime(FIXED_START_DATE)
        if self.rebalancing_date < FIXED_START_DATE_DT:
            raise ValueError(
                f"Rebalancing date {self.rebalancing_date.date()} is before FIXED_START_DATE {FIXED_START_DATE_DT.date()}."
            )

        self.logger.debug(
            f"Initializing EconomicIndicator for '{self.name}' with rebalancing date {self.rebalancing_date.date()}."
        )
        self.process_data(data)

        # Map calculation methods to their respective functions
        self.calculation_method_functions: Dict[
            CalculationMethod, Callable[..., Optional[float]]
        ] = {
            CalculationMethod.PERCENTAGE_CHANGE: self._calculate_percentage_change,
            CalculationMethod.BASIS_POINTS_CHANGE: self._calculate_basis_points_change,
            CalculationMethod.Z_SCORE: self._calculate_z_score,
            CalculationMethod.ABSOLUTE_CHANGE: self._calculate_absolute_change,
            CalculationMethod.YEAR_OVER_YEAR_CHANGE: self._calculate_year_over_year_change,
            CalculationMethod.CAGR: self._calculate_cagr,
            CalculationMethod.CURRENT_VALUE: self._use_current_value,
        }

    def process_data(self, data_points: List[Dict[str, Any]]) -> None:
        """
        Process raw data points into a cleaned DataFrame.

        Converts raw data into a Pandas DataFrame, ensuring proper date and value formatting.
        This method also filters data up to the rebalancing date and sorts it in reverse chronological order
        to facilitate analysis.

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

        self.logger.debug(
            f"Processing data for '{self.name}' up to {self.rebalancing_date.date()}."
        )

        # Convert 'date' to datetime and 'value' to numeric, coercing errors to NaN
        original_df["date"] = pd.to_datetime(original_df["date"], errors="coerce")
        original_df["value"] = pd.to_numeric(original_df["value"], errors="coerce")

        # Filter data up to the rebalancing date
        cleaned_df = original_df.copy()
        cleaned_df = cleaned_df[cleaned_df["date"] <= self.rebalancing_date]
        cleaned_df.set_index("date", inplace=True)
        cleaned_df.sort_index(ascending=False, inplace=True)

        self.data = cleaned_df
        self.logger.info(
            f"Processed data for '{self.name}' with {len(cleaned_df)} records up to {self.rebalancing_date.date()}, including NaN values for alignment."
        )

    def extract_time_frame_data(self, years: int) -> Optional[pd.DataFrame]:
        """
        Extract data for a specific time frame.

        Filters the data to include only records within the specified number of years from the rebalancing date.

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

        end_date: pd.Timestamp = self.rebalancing_date

        # Compute start_date, but ensure it's not before the earliest date of the data
        earliest_data_date: pd.Timestamp = self.data.index.min()
        requested_start_date: pd.Timestamp = end_date - pd.DateOffset(years=years)
        start_date: pd.Timestamp = max(requested_start_date, earliest_data_date)

        if start_date > end_date:
            self.logger.warning(
                f"No data available between {start_date.date()} and {end_date.date()} for indicator '{self.name}'."
            )
            return None

        time_frame_data: pd.DataFrame = self.data[
            (self.data.index >= start_date) & (self.data.index <= end_date)
        ]

        if time_frame_data.empty:
            self.logger.warning(
                f"No data available for {years}-year period up to {end_date.date()} for indicator '{self.name}'."
            )
            return None

        self.logger.debug(
            f"Extracted {len(time_frame_data)} records for {years}-year period up to {end_date.date()} for '{self.name}'."
        )
        return time_frame_data

    def calculate_statistics(self, data: pd.Series) -> Dict[str, Optional[float]]:
        """
        Calculate basic statistics for a given data series.

        Computes metrics such as the most recent value, average, median, minimum, maximum,
        standard deviation, and z-score to understand the data distribution.

        Args:
            data (pd.Series): Series of numerical values representing the economic indicator.

        Returns:
            Dict[str, Optional[float]]: Dictionary containing calculated statistics.

        Example:
            >>> stats = cpi_indicator.calculate_statistics(cpi_data["value"])
        """
        self.logger.debug(f"Calculating statistics for '{self.name}'.")

        # Remove NaN values to ensure accurate calculations
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

        most_recent_value = data.iloc[0]
        average = data.mean()
        median = data.median()
        min_value = data.min()
        max_value = data.max()
        std_dev = data.std()
        z_score = self.calculate_z_score(data)

        stats = {
            "most_recent_value": most_recent_value,
            "average": average,
            "median": median,
            "min": min_value,
            "max": max_value,
            "std_dev": std_dev,
            "z_score": z_score,
        }
        self.logger.debug(f"Statistics for '{self.name}': {stats}")
        return stats

    def calculate_z_score(self, series: pd.Series) -> Optional[float]:
        """
        Calculate the z-score for the most recent value in the series.

        The z-score indicates how many standard deviations the most recent value is from the mean.

        Args:
            series (pd.Series): Series of numerical values.

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

        mean = series.mean()
        std_dev = series.std()

        if std_dev == 0.0:
            self.logger.debug(
                f"Standard deviation is zero for '{self.name}', cannot calculate z-score."
            )
            return None  # Avoid division by zero

        value = float(series.iloc[0])
        z_score = (value - mean) / std_dev
        self.logger.debug(f"Z-score for '{self.name}': {z_score:.2f}")
        return z_score

    def _calculate_percentage_change(
        self, current_value: float, start_value: float, **kwargs: Any
    ) -> Optional[float]:
        """
        Calculate percentage change.

        Args:
            current_value (float): Current value of the indicator.
            start_value (float): Value of the indicator at the start of the period.

        Returns:
            Optional[float]: Percentage change, or None if start_value is zero.
        """
        if start_value == 0.0:
            self.logger.warning(
                f"Cannot calculate percentage change for '{self.name}': start_value is zero."
            )
            return None
        change = ((current_value - start_value) / start_value) * 100.0
        self.logger.debug(f"Percentage change for '{self.name}': {change:.2f}%.")
        return change

    def _calculate_basis_points_change(
        self, current_value: float, start_value: float, **kwargs: Any
    ) -> Optional[float]:
        """
        Calculate change in basis points.

        Args:
            current_value (float): Current value of the indicator.
            start_value (float): Value of the indicator at the start of the period.

        Returns:
            float: Change in basis points.
        """
        change = (current_value - start_value) * 100  # Basis points
        self.logger.debug(f"Basis points change for '{self.name}': {change:.2f} bps.")
        return change

    def _calculate_absolute_change(
        self, current_value: float, start_value: float, **kwargs: Any
    ) -> Optional[float]:
        """
        Calculate absolute change.

        Args:
            current_value (float): Current value of the indicator.
            start_value (float): Value of the indicator at the start of the period.

        Returns:
            float: Absolute change.
        """
        change = current_value - start_value
        self.logger.debug(f"Absolute change for '{self.name}': {change}")
        return change

    def _calculate_z_score(
        self, data_series: pd.Series, **kwargs: Any
    ) -> Optional[float]:
        """
        Calculate the z-score for the data series.

        Args:
            data_series (pd.Series): Series of numerical values.

        Returns:
            Optional[float]: Z-score, or None if calculation is not possible.
        """
        return self.calculate_z_score(data_series)

    def _calculate_year_over_year_change(
        self, data_series: pd.Series, **kwargs: Any
    ) -> Optional[float]:
        """
        Calculate the year-over-year (YoY) change.

        Args:
            data_series (pd.Series): Series of numerical values indexed by date.

        Returns:
            Optional[float]: YoY change as a percentage, or None if calculation is not possible.
        """
        # Remove NaN values to ensure accurate calculations
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

        # Sort the series in ascending order for accurate date matching
        sorted_series = data_series.sort_index(ascending=True)

        # Find the closest available date using nearest method
        indexer = sorted_series.index.get_indexer([one_year_ago_date], method="nearest")
        pos = indexer[0]

        if pos == -1:
            self.logger.warning(f"Cannot find data one year ago for '{self.name}'.")
            return None

        value_one_year_ago_raw = sorted_series.iloc[pos]

        # Attempt to cast values to float
        try:
            value_one_year_ago: float = float(value_one_year_ago_raw)
            current_value: float = float(data_series.iloc[0])  # Original series
        except (ValueError, TypeError) as e:
            self.logger.error(f"Error casting values to float: {e}")
            return None

        # Check for NaN values after casting
        if pd.isna(current_value) or pd.isna(value_one_year_ago):
            self.logger.warning(
                f"Cannot calculate year-over-year change for '{self.name}': current or year-ago value is NaN."
            )
            return None

        # Prevent division by zero
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

    def _calculate_cagr(
        self, data_series: pd.Series, periods_per_year: int, **kwargs: Any
    ) -> Optional[float]:
        """
        Calculate the Compound Annual Growth Rate (CAGR) for the indicator.

        Args:
            data_series (pd.Series): Series of numerical values indexed by date.
            periods_per_year (int): Number of periods per year based on indicator frequency.

        Returns:
            Optional[float]: CAGR as a percentage, or None if calculation is not possible.
        """
        # Remove NaN values to ensure accurate calculations
        data = data_series.dropna()

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
            # Ensure n_years is a valid float greater than zero
            cagr: float = (end_value / start_value) ** (1.0 / n_years) - 1.0
            cagr_percentage = cagr * 100.0
            self.logger.debug(f"CAGR for '{self.name}': {cagr_percentage:.2f}%")
            return cagr_percentage
        except (ZeroDivisionError, OverflowError) as e:
            self.logger.error(f"CAGR calculation error for '{self.name}': {e}")
            return None

    def _use_current_value(
        self, current_value: float, start_value: float, **kwargs: Any
    ) -> Optional[float]:
        """
        Use the current value directly as the change.

        Args:
            current_value (float): Current value of the indicator.
            start_value (float): Start value (unused).

        Returns:
            float: Current value.
        """
        self.logger.debug(f"Current value for '{self.name}': {current_value}")
        return current_value

    def calculate_change(
        self,
        current_value: Optional[float],
        start_value: Optional[float],
        data_series: pd.Series,
        periods_per_year: int,
    ) -> Optional[float]:
        """
        Calculate the change based on the specified calculation method.

        Utilizes a mapping of calculation methods to their respective functions.

        Args:
            current_value (Optional[float]): The most recent value of the indicator.
            start_value (Optional[float]): The value of the indicator at the start of the period.
            data_series (pd.Series): Series of historical data points.
            periods_per_year (int): Number of periods per year based on indicator frequency.

        Returns:
            Optional[float]: Calculated change value, or None if calculation is not possible.

        Example:
            >>> change = cpi_indicator.calculate_change(250.5, 245.0, cpi_data["value"], 12)
        """
        method_function = self.calculation_method_functions.get(
            self.config.calculation_method
        )
        if not method_function:
            self.logger.warning(
                f"Unknown calculation method '{self.config.calculation_method}' for '{self.name}'."
            )
            return None

        if self.config.calculation_method in {
            CalculationMethod.PERCENTAGE_CHANGE,
            CalculationMethod.BASIS_POINTS_CHANGE,
            CalculationMethod.ABSOLUTE_CHANGE,
            CalculationMethod.CURRENT_VALUE,
        }:
            if current_value is None or start_value is None:
                self.logger.warning(
                    f"Current or start value is None for '{self.name}'. Cannot compute change."
                )
                return None
            return method_function(current_value, start_value)
        elif self.config.calculation_method == CalculationMethod.Z_SCORE:
            return method_function(data_series)
        elif self.config.calculation_method == CalculationMethod.YEAR_OVER_YEAR_CHANGE:
            return method_function(data_series)
        elif self.config.calculation_method == CalculationMethod.CAGR:
            return method_function(data_series, periods_per_year)
        else:
            self.logger.warning(
                f"Unhandled calculation method '{self.config.calculation_method}' for '{self.name}'."
            )
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

        upper_threshold, lower_threshold = self.config.thresholds
        self.logger.debug(
            f"Generating trend signal for '{self.name}' with change: {change} "
            f"and thresholds: upper={upper_threshold}, lower={lower_threshold}."
        )

        # Determine trend based on calculation method thresholds
        if self.config.calculation_method in {
            CalculationMethod.YEAR_OVER_YEAR_CHANGE,
            CalculationMethod.CAGR,
            CalculationMethod.CURRENT_VALUE,
        }:
            # Thresholds are positive numbers for these methods
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
                            weighted changes, statistics, and the rebalancing date.

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

        # Determine the number of periods per year based on indicator frequency
        frequency: IndicatorType = self.indicator_type
        if frequency == IndicatorType.QUARTERLY:
            periods_per_year = 4
        elif frequency == IndicatorType.MONTHLY:
            periods_per_year = 12
        elif frequency == IndicatorType.DAILY:
            periods_per_year = 252  # Approximate trading days in a year
        else:
            periods_per_year = 1  # Default to annual

        self.logger.debug(
            f"Analyzing '{self.name}' with frequency '{frequency.value}' and periods_per_year={periods_per_year}."
        )

        for tf_weight in self.time_frame_weights:
            period_years = tf_weight.years
            weight = tf_weight.weight
            self.logger.debug(
                f"Analyzing '{self.name}' for {period_years}-year period with weight {weight}."
            )
            time_frame_data: Optional[pd.DataFrame] = self.extract_time_frame_data(
                period_years
            )
            if time_frame_data is None or time_frame_data.empty:
                continue

            values: pd.Series = time_frame_data["value"].dropna()
            if values.empty:
                self.logger.warning(
                    f"No valid data points for {period_years}-year period for '{self.name}'."
                )
                continue

            current_value: Optional[float] = (
                float(values.iloc[0]) if not values.empty else None
            )

            start_value: Optional[float] = (
                float(values.iloc[-1]) if len(values) > 1 else None
            )

            if current_value is not None:
                # Calculate change based on the specified method
                change = self.calculate_change(
                    current_value, start_value, values, periods_per_year
                )
            else:
                change = None
                self.logger.warning(
                    f"Current value is None for '{self.name}' in {period_years}-year period."
                )

            # Generate trend signal using thresholds
            signal: str = self.generate_trend_signal(change)

            # Collect additional metrics if needed
            year_over_year_change = (
                self._calculate_year_over_year_change(values)
                if self.config.calculation_method
                == CalculationMethod.YEAR_OVER_YEAR_CHANGE
                else None
            )

            cagr = (
                self._calculate_cagr(values, periods_per_year)
                if self.config.calculation_method == CalculationMethod.CAGR
                else None
            )

            z_score = (
                self.calculate_z_score(values)
                if self.config.calculation_method == CalculationMethod.Z_SCORE
                else None
            )

            # Store analysis results for the current time frame
            results[f"{period_years}y"] = {
                "current_value": current_value,
                "change": change,
                "cagr": cagr,
                "signal": signal,
                "z_score": z_score,
                "year_over_year_change": year_over_year_change,
            }

            self.logger.debug(
                f"Analysis for {period_years}y period of '{self.name}': {results[f'{period_years}y']}"
            )

            # Accumulate weighted changes for overall analysis
            if change is not None:
                if not math.isnan(change):
                    weighted_change_sum += change * weight
                    total_weight += weight

            # Record trend direction for overall trend determination
            if signal == "Rising":
                trend_directions.append(1)
            elif signal == "Falling":
                trend_directions.append(-1)
            else:
                trend_directions.append(0)

        # Calculate weighted change if applicable
        weighted_change = (
            weighted_change_sum / total_weight if total_weight > 0 else None
        )
        results["weighted"] = {
            "weighted_change": weighted_change,
        }
        self.logger.debug(f"Weighted change for '{self.name}': {weighted_change}")

        if total_weight == 0.0:
            self.logger.warning(
                f"No data available for any time frames for '{self.name}'."
            )
            results["overall_trend"] = "No Data"
            return results

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
        stats: Dict[str, Optional[float]] = self.calculate_statistics(
            self.data["value"]
        )
        results["statistics"] = stats
        self.logger.debug(f"Statistics for '{self.name}': {stats}")

        # Include Rebalancing Date in Results
        results["rebalancing_date"] = self.rebalancing_date.strftime("%Y-%m-%d")
        self.logger.debug(
            f"Rebalancing date for '{self.name}': {results['rebalancing_date']}"
        )

        self.logger.info(f"Completed analysis for '{self.name}'.")
        return results

    def analyze_all_indicators(self) -> Dict[str, Any]:
        """
        Analyze the economic indicator and return the analysis results.

        This method serves as a wrapper to perform the complete analysis process, encapsulating
        data processing, statistical calculations, and trend signal generation.

        Returns:
            Dict[str, Any]: Comprehensive analysis results for the indicator, including the rebalancing date.

        Example:
            >>> analysis_results = cpi_indicator.analyze_all_indicators()
        """
        self.logger.debug(f"Starting comprehensive analysis for '{self.name}'.")
        analysis = self.analyze_indicator()
        self.logger.debug(
            f"Comprehensive analysis results for '{self.name}': {analysis}"
        )
        return analysis
