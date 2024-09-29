# src/config.py

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import List, Tuple


class IndicatorType(Enum):
    DAILY = "DAILY"
    MONTHLY = "MONTHLY"
    QUARTERLY = "QUARTERLY"


class CalculationMethod(Enum):
    PERCENTAGE_CHANGE = "percentage_change"
    BASIS_POINTS_CHANGE = "basis_points_change"
    Z_SCORE = "z_score"
    ABSOLUTE_CHANGE = "absolute_change"
    YEAR_OVER_YEAR_CHANGE = "year_over_year_change"
    CAGR = "cagr"
    CURRENT_VALUE = "current_value"


@dataclass
class TimeFrameWeight:
    years: int
    weight: float


@dataclass
class IndicatorConfig:
    """
    Configuration for an economic indicator.

    Attributes:
        series_id (str): The unique identifier for the data series as per the data source.
        name (str): A human-readable name for the economic indicator.
        internal_key (str): A key used internally within the application to reference the indicator.
        description (str): A brief description of the economic indicator.
        indicator_type (IndicatorType): The frequency type of the indicator data (e.g., 'MONTHLY', 'DAILY', 'QUARTERLY').
        time_frame_weights (List[TimeFrameWeight]):
            A list of TimeFrameWeight instances mapping time frames (in years) to their respective weights.
            These weights determine the importance of each time frame during analysis.
        calculation_method (CalculationMethod):
            The method used to calculate the change for this indicator.
        thresholds (Tuple[float, float]):
            A tuple containing the upper and lower threshold values specific to the calculation method for generating trend signals.
        rule_weight (float):
            A weight assigned to the rule associated with this indicator.
            Determines the significance of this rule in portfolio adjustments.
            Higher weights imply greater impact.
        earliest_date (str):
            The earliest available date for the indicator data in "YYYY-MM-DD" format.
    """

    series_id: str
    name: str
    internal_key: str
    description: str
    indicator_type: IndicatorType
    time_frame_weights: List[TimeFrameWeight]
    calculation_method: CalculationMethod
    thresholds: Tuple[float, float]
    rule_weight: float
    earliest_date: str


# Centralized list of all economic indicators with their configurations
INDICATORS: List[IndicatorConfig] = [
    # Federal Funds Rate
    IndicatorConfig(
        series_id="FEDFUNDS",
        name="Federal Funds Rate",
        internal_key="fed_funds",
        description="Federal Funds Rate",
        indicator_type=IndicatorType.MONTHLY,
        time_frame_weights=[
            TimeFrameWeight(years=1, weight=0.30),
            TimeFrameWeight(years=3, weight=0.40),
            TimeFrameWeight(years=5, weight=0.30),
        ],
        calculation_method=CalculationMethod.Z_SCORE,
        thresholds=(1.0, -1.0),  # Z-score thresholds for trend signals
        rule_weight=1.0,  # Highest priority
        earliest_date="1954-12-01",
    ),
    # Consumer Price Index (CPI)
    IndicatorConfig(
        series_id="CPIAUCSL",
        name="Consumer Price Index (CPI)",
        internal_key="cpi",
        description="Composite CPI & PCE Inflation",
        indicator_type=IndicatorType.MONTHLY,
        time_frame_weights=[
            TimeFrameWeight(years=1, weight=1.0),  # Focus on 1-year for YoY change
        ],
        calculation_method=CalculationMethod.YEAR_OVER_YEAR_CHANGE,
        thresholds=(2.5, 2.0),  # Inflation targets (upper, lower)
        rule_weight=0.9,
        earliest_date="1913-03-01",
    ),
    # Personal Consumption Expenditures Price Index (PCE)
    IndicatorConfig(
        series_id="PCEPI",
        name="Personal Consumption Expenditures Price Index (PCE)",
        internal_key="pce",
        description="Composite CPI & PCE Inflation",
        indicator_type=IndicatorType.MONTHLY,
        time_frame_weights=[
            TimeFrameWeight(years=1, weight=1.0),
        ],
        calculation_method=CalculationMethod.YEAR_OVER_YEAR_CHANGE,
        thresholds=(2.5, 2.0),  # Inflation targets
        rule_weight=0.9,
        earliest_date="1947-04-01",
    ),
    # 5-Year Breakeven Inflation Rate
    IndicatorConfig(
        series_id="T5YIE",
        name="5-Year Breakeven Inflation Rate",
        internal_key="breakeven_inflation",
        description="Breakeven Inflation",
        indicator_type=IndicatorType.DAILY,
        time_frame_weights=[
            TimeFrameWeight(years=1, weight=1.0),  # Focus on recent data
        ],
        calculation_method=CalculationMethod.BASIS_POINTS_CHANGE,
        thresholds=(10, -10),  # Thresholds in basis points
        rule_weight=0.8,
        earliest_date="2006-01-01",
    ),
    # Yield Spread
    IndicatorConfig(
        series_id="T10Y2Y",
        name="Yield Spread",
        internal_key="yield_spread",
        description="Yield Spread",
        indicator_type=IndicatorType.DAILY,
        time_frame_weights=[
            TimeFrameWeight(years=1, weight=0.20),
            TimeFrameWeight(years=3, weight=0.50),
            TimeFrameWeight(years=5, weight=0.30),
        ],
        calculation_method=CalculationMethod.BASIS_POINTS_CHANGE,
        thresholds=(50, -50),  # Thresholds in basis points
        rule_weight=0.7,
        earliest_date="1990-01-01",
    ),
    # Recession Probabilities
    IndicatorConfig(
        series_id="RECPROUSM156N",
        name="Recession Probabilities",
        internal_key="recession_prob",
        description="Recession Probability",
        indicator_type=IndicatorType.MONTHLY,
        time_frame_weights=[
            TimeFrameWeight(years=1, weight=1.0),  # Focus on recent data
        ],
        calculation_method=CalculationMethod.CURRENT_VALUE,
        thresholds=(25.0, 15.0),  # Thresholds in percentage
        rule_weight=0.6,
        earliest_date="1984-01-01",
    ),
    # Credit Spread
    IndicatorConfig(
        series_id="BAA10YM",
        name="Moody's BAA Corporate Bond Yield Minus 10-Year Treasury Yield",
        internal_key="credit_spread",
        description="Credit Spread",
        indicator_type=IndicatorType.MONTHLY,
        time_frame_weights=[
            TimeFrameWeight(years=1, weight=0.25),
            TimeFrameWeight(years=3, weight=0.50),
            TimeFrameWeight(years=5, weight=0.25),
        ],
        calculation_method=CalculationMethod.BASIS_POINTS_CHANGE,
        thresholds=(50, -50),  # Thresholds in basis points
        rule_weight=0.5,
        earliest_date="1994-01-01",
    ),
    # Real Gross Domestic Product (GDP)
    IndicatorConfig(
        series_id="GDPC1",
        name="Real Gross Domestic Product (GDP)",
        internal_key="gdp",
        description="GDP Growth Rate",
        indicator_type=IndicatorType.QUARTERLY,
        time_frame_weights=[
            TimeFrameWeight(years=1, weight=1.0),
        ],
        calculation_method=CalculationMethod.CAGR,
        thresholds=(3.0, 2.0),  # Thresholds in percentage points
        rule_weight=0.4,
        earliest_date="1947-01-01",
    ),
    # Unemployment Rate
    IndicatorConfig(
        series_id="UNRATE",
        name="Unemployment Rate",
        internal_key="unrate",
        description="Unemployment Rate",
        indicator_type=IndicatorType.MONTHLY,
        time_frame_weights=[
            TimeFrameWeight(years=1, weight=1.0),  # Focus on recent changes
        ],
        calculation_method=CalculationMethod.ABSOLUTE_CHANGE,
        thresholds=(0.5, -0.5),  # Thresholds in percentage points
        rule_weight=0.3,  # Lowest priority
        earliest_date="1948-01-01",
    ),
]


# Programmatically compute FIXED_START_DATE as the maximum of all earliest_date values
def compute_fixed_start_date(indicators: List[IndicatorConfig]) -> str:
    """
    Compute the FIXED_START_DATE as the latest earliest_date among all indicators.

    Returns:
        str: The FIXED_START_DATE in "YYYY-MM-DD" format.
    """
    earliest_dates = [
        datetime.strptime(indicator.earliest_date, "%Y-%m-%d")
        for indicator in indicators
    ]
    if not earliest_dates:
        raise ValueError(
            "No earliest_date found in indicators to compute FIXED_START_DATE."
        )

    fixed_start_date = max(earliest_dates)
    return fixed_start_date.strftime("%Y-%m-%d")


FIXED_START_DATE: str = compute_fixed_start_date(INDICATORS)
