# src/config.py

from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class IndicatorConfig:
    """
    Configuration for an economic indicator.

    Attributes:
        series_id (str): The unique identifier for the data series as per the data source.
        name (str): A human-readable name for the economic indicator.
        internal_key (str): A key used internally within the application to reference the indicator.
        indicator_type (str): The frequency type of the indicator data (e.g., 'MONTHLY', 'DAILY', 'QUARTERLY').
        time_frame_weights (Dict[int, float]):
            A dictionary mapping time frames (in years) to their respective weights.
            These weights determine the importance of each time frame during analysis.
        calculation_method (str):
            The method used to calculate the change for this indicator.
            Possible values include:
            - `'percentage_change'`
            - `'basis_points_change'`
            - `'z_score'`
            - `'absolute_change'`
            - `'year_over_year_change'`
            - `'cagr'`
            - `'current_value'`
        thresholds (Tuple[float, float]):
            A tuple containing the upper and lower threshold values specific to the calculation method for generating trend signals.
        rule_weight (float):
            A weight assigned to the rule associated with this indicator.
            Determines the significance of this rule in portfolio adjustments.
            Higher weights imply greater impact.
    """

    series_id: str
    name: str
    internal_key: str
    indicator_type: str
    time_frame_weights: Dict[int, float]
    calculation_method: str
    thresholds: Tuple[float, float]
    rule_weight: float  # New attribute for rule weighting


# Centralized list of all economic indicators with their configurations
INDICATORS: List[IndicatorConfig] = [
    # Federal Funds Rate
    IndicatorConfig(
        series_id="FEDFUNDS",
        name="Federal Funds Rate",
        internal_key="fed_funds",
        indicator_type="MONTHLY",
        time_frame_weights={1: 0.30, 3: 0.40, 5: 0.30},
        calculation_method="z_score",
        thresholds=(1.5, -1.5),  # Z-score thresholds for trend signals
        rule_weight=1.0,  # Highest priority
    ),
    # Consumer Price Index (CPI)
    IndicatorConfig(
        series_id="CPIAUCSL",
        name="Consumer Price Index (CPI)",
        internal_key="cpi",
        indicator_type="MONTHLY",
        time_frame_weights={1: 1.0},  # Focus on 1-year for YoY change
        calculation_method="year_over_year_change",
        thresholds=(2.5, 2.0),  # Inflation targets (upper, lower)
        rule_weight=0.9,
    ),
    # Personal Consumption Expenditures Price Index (PCE)
    IndicatorConfig(
        series_id="PCEPI",
        name="Personal Consumption Expenditures Price Index (PCE)",
        internal_key="pce",
        indicator_type="MONTHLY",
        time_frame_weights={1: 1.0},
        calculation_method="year_over_year_change",
        thresholds=(2.5, 2.0),  # Inflation targets
        rule_weight=0.9,
    ),
    # Yield Spread
    IndicatorConfig(
        series_id="T10Y2Y",
        name="Yield Spread",
        internal_key="yield_spread",
        indicator_type="DAILY",
        time_frame_weights={1: 0.20, 3: 0.50, 5: 0.30},
        calculation_method="basis_points_change",
        thresholds=(50, -50),  # Thresholds in basis points
        rule_weight=0.7,
    ),
    # Unemployment Rate
    IndicatorConfig(
        series_id="UNRATE",
        name="Unemployment Rate",
        internal_key="unrate",
        indicator_type="MONTHLY",
        time_frame_weights={1: 1.0},  # Focus on recent changes
        calculation_method="absolute_change",
        thresholds=(0.5, -0.5),  # Thresholds in percentage points
        rule_weight=0.3,  # Lowest priority
    ),
    # Real Gross Domestic Product (GDP)
    IndicatorConfig(
        series_id="GDPC1",
        name="Real Gross Domestic Product (GDP)",
        internal_key="gdp",
        indicator_type="QUARTERLY",
        time_frame_weights={1: 1.0},
        calculation_method="cagr",
        thresholds=(3.0, 2.0),  # Thresholds in percentage points
        rule_weight=0.4,
    ),
    # 5-Year Breakeven Inflation Rate
    IndicatorConfig(
        series_id="T5YIE",
        name="5-Year Breakeven Inflation Rate",
        internal_key="breakeven_inflation",
        indicator_type="DAILY",
        time_frame_weights={1: 1.0},  # Focus on recent data
        calculation_method="basis_points_change",
        thresholds=(10, -10),  # Thresholds in basis points
        rule_weight=0.8,
    ),
    # Recession Probabilities
    IndicatorConfig(
        series_id="RECPROUSM156N",
        name="Recession Probabilities",
        internal_key="recession_prob",
        indicator_type="MONTHLY",
        time_frame_weights={1: 1.0},  # Focus on recent data
        calculation_method="current_value",
        thresholds=(25.0, 15.0),  # Thresholds in percentage
        rule_weight=0.6,
    ),
    # Credit Spread
    IndicatorConfig(
        series_id="BAA10YM",
        name="Moody's BAA Corporate Bond Yield Minus 10-Year Treasury Yield",
        internal_key="credit_spread",
        indicator_type="MONTHLY",
        time_frame_weights={1: 0.25, 3: 0.50, 5: 0.25},
        calculation_method="basis_points_change",
        thresholds=(50, -50),  # Thresholds in basis points
        rule_weight=0.5,
    ),
]
