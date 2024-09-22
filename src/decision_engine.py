# src/decision_engine.py

"""
decision_engine.py

This module defines the `DecisionEngine` class, responsible for analyzing economic indicators
and adjusting the investment portfolio accordingly. It applies various decision rules
based on economic trends to optimize the portfolio's allocations, ensuring alignment with strategic
objectives and risk constraints.

Classes:
    DecisionEngine: Core class that applies decision rules to manage and adjust the investment portfolio.
"""

import datetime
import logging
import math
from typing import Any, Dict, Optional

from .decision_rules.decision_rules import DecisionRules
from .indicators import EconomicIndicator
from .portfolio import Portfolio


class DecisionEngine:
    """
    Core engine for making investment decisions based on economic indicators.

    The `DecisionEngine` analyzes various economic indicators to determine appropriate adjustments
    to the investment portfolio. It applies a series of decision rules to optimize asset allocations,
    ensuring the portfolio remains aligned with strategic objectives and risk constraints.

    Attributes:
        indicators (Dict[str, EconomicIndicator]): A dictionary of economic indicators used for analysis.
        portfolio (Portfolio): The investment portfolio to be managed and adjusted.
        rule_messages (Dict[str, List[str]]): Messages associated with each decision rule applied.
        max_adjustment (float): Maximum percentage adjustment allowed for any operation.
        logger (logging.Logger): Logger instance for logging activities and debugging.
        initializing (bool): Flag indicating whether the engine is in the initialization phase.
        baseline_allocations (Dict[str, float]): Strategic baseline allocations for portfolio assets.
        asset_risk_levels (Dict[str, float]): Risk levels assigned to each asset class.
        max_portfolio_risk (float): Maximum acceptable average risk level for the portfolio.
        analysis_results (Dict[str, Dict[str, Any]]): Results from analyzing each economic indicator.
    """

    def __init__(
        self,
        indicators: Dict[str, EconomicIndicator],
        portfolio: Portfolio,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        """
        Initialize the DecisionEngine with economic indicators and the investment portfolio.

        Args:
            indicators (Dict[str, EconomicIndicator]):
                Dictionary of EconomicIndicator objects used for portfolio analysis.
            portfolio (Portfolio):
                The Portfolio object to adjust based on economic indicators.
            logger (Optional[logging.Logger], optional):
                Logger instance for logging messages. If None, a default logger is used.

        Raises:
            ValueError: If baseline allocations do not sum to 100%.
        """
        self.indicators: Dict[str, EconomicIndicator] = indicators
        self.portfolio: Portfolio = portfolio
        self.rule_messages: Dict[str, list] = {}
        self.max_adjustment: float = (
            10.0  # Maximum adjustment percentage for all operations
        )
        self.logger: logging.Logger = logger or logging.getLogger(
            "Bondit.DecisionEngine"
        )

        self.initializing: bool = False  # Flag to indicate initialization phase

        # Define strategic baseline allocations (total should sum to 100%)
        self.baseline_allocations: Dict[str, float] = {
            "Government Bonds - Long-Term": 30.0,
            "Corporate Bonds - Long-Term": 20.0,
            "Government Bonds - Intermediate-Term": 20.0,
            "Corporate Bonds - Intermediate-Term": 15.0,
            "Short-Term Bonds": 10.0,
            "TIPS": 5.0,
            "Nominal Bonds": 0.0,
        }

        if not math.isclose(
            sum(self.baseline_allocations.values()), 100.0, abs_tol=0.1
        ):
            self.logger.critical(
                "Baseline allocations do not sum to 100%. Initialization aborted."
            )
            raise ValueError("Baseline allocations must sum to 100%.")

        # Define asset risk levels (higher number indicates higher risk)
        self.asset_risk_levels: Dict[str, float] = {
            "Government Bonds - Long-Term": 2.0,
            "Corporate Bonds - Long-Term": 3.0,
            "Government Bonds - Intermediate-Term": 2.0,
            "Corporate Bonds - Intermediate-Term": 3.0,
            "Short-Term Bonds": 1.0,
            "TIPS": 2.0,
            "Nominal Bonds": 2.5,
        }

        # Set maximum acceptable average portfolio risk level
        self.max_portfolio_risk: float = 2.5

        # Initialize analysis results
        self.analysis_results: Dict[str, Dict[str, Any]] = {}
        self.logger.debug("Initializing DecisionEngine.")
        for indicator_name, indicator in self.indicators.items():
            self.logger.debug(f"Analyzing indicator '{indicator_name}'.")
            self.analysis_results[indicator_name] = indicator.analyze_indicator()
        self.logger.info("DecisionEngine initialized with analysis results.")

    def get_rule_weight(self, rule_key: str) -> float:
        """
        Retrieve the weight for a given rule based on IndicatorConfig.

        Args:
            rule_key (str): The internal key of the rule (matches IndicatorConfig.internal_key).

        Returns:
            float: The weight associated with the rule.
        """
        indicator = self.indicators.get(rule_key)
        if indicator and hasattr(indicator.config, "rule_weight"):
            # Assuming 'rule_weight' is a float attribute of 'config'
            return indicator.config.rule_weight
        else:
            self.logger.warning(
                f"No rule weight found for '{rule_key}'. Using default weight of 1.0."
            )
            return 1.0  # Default weight if not specified

    def apply_decision_rules(self) -> None:
        """
        Apply all decision rules to adjust the portfolio based on economic indicators.

        This method determines whether the portfolio is in an initialization phase or a regular
        adjustment phase. It then applies each decision rule accordingly and adjusts the portfolio
        allocations to align with strategic objectives and risk constraints. After adjustments,
        it generates a rebalancing report summarizing the actions taken.
        """
        self.logger.info("Applying decision rules.")
        self.rule_messages = {}

        if self.is_empty():
            self.logger.info(
                "Portfolio is empty. Initializing with baseline allocations."
            )
            self.initialize_allocations()
        else:
            self.logger.info("Applying adjustments based on decision rules.")
            DecisionRules.apply_all_rules(self)

            self.logger.info("Rebalancing portfolio after adjustments.")
            self.portfolio.rebalance()

        self.logger.info("Decision rules applied and portfolio adjusted.")

    def initialize_allocations(self) -> None:
        """
        Initialize the portfolio allocations based on the baseline allocations and apply
        decision rules for initial adjustments.
        """
        self.logger.debug("Initializing allocations from baseline.")
        self.portfolio.set_allocations(self.baseline_allocations)
        DecisionRules.apply_all_rules(self)
        self.portfolio.rebalance()

    def adjust_allocation(
        self, asset: str, amount: float, rule: str, rule_weight: float
    ) -> None:
        """
        Adjust the allocation for a specific asset based on a decision rule.

        Args:
            asset (str): The asset to adjust.
            amount (float): The adjustment amount (positive to increase, negative to decrease).
            rule (str): The decision rule that proposes this adjustment.
            rule_weight (float): The weight of the decision rule.
        """
        # Limit adjustment to the maximum allowed
        if abs(amount) > self.max_adjustment:
            adjusted_amount = math.copysign(self.max_adjustment, amount)
            self.logger.warning(
                f"Adjustment for '{asset}' by {amount:.2f}% exceeds max adjustment. Capped to {adjusted_amount:.2f}%."
            )
            amount = adjusted_amount

        self.portfolio.adjust_allocation(asset, amount, rule, rule_weight)
        message = f"{self.get_rule_description(rule)}: {'Increased' if amount > 0 else 'Decreased'} '{asset}' by {abs(amount):.2f}%."
        self._add_rule_message(rule, message)
        self.logger.info(message)

    def generate_rebalancing_report(self) -> str:
        """
        Generate a plain text rebalancing report based on current economic indicators,
        portfolio adjustments, and updated portfolio allocations.

        Returns:
            str: A string containing the formatted plain text report.
        """
        report_lines = []
        report_date = datetime.datetime.now().strftime("%Y-%m-%d")

        # Report Header
        report_lines.append(f"Rebalancing Report Date: {report_date}\n")

        # Rebalancing Scenario Section
        report_lines.append("Rebalancing Scenario")
        report_lines.append("---------------------\n")

        # Portfolio Adjustments
        report_lines.append("Portfolio Adjustments:")
        if not self.rule_messages:
            report_lines.append("  - No adjustments made.\n")
        else:
            # Iterate through the rule_messages to provide context
            for rule, messages in self.rule_messages.items():
                for message in messages:
                    report_lines.append(f"  - {message}\n")

        # Updated Portfolio Allocations
        report_lines.append("\n**Updated Portfolio Allocations:**")
        for asset, allocation in self.portfolio.get_allocations().items():
            report_lines.append(f"  - {asset}: {allocation:.2f}%")

        # Combine all lines into a single string
        return "\n".join(report_lines)

    def save_rebalancing_report(self, file_path: str = "rebalancing_report.md") -> None:
        """
        Save the generated rebalancing report to a markdown file.

        Args:
            file_path (str, optional): The path to the file where the report will be saved.
                                       Defaults to "rebalancing_report.md".

        Raises:
            IOError: If the file cannot be written.
        """
        report = self.generate_rebalancing_report()
        try:
            with open(file_path, "w") as file:
                file.write(report)
            self.logger.info(f"Rebalancing report saved to {file_path}.")
        except Exception as e:
            self.logger.error(f"Failed to save rebalancing report to {file_path}: {e}")

    def get_rule_description(self, rule: str) -> str:
        """
        Retrieve a human-readable description for a given decision rule.

        Args:
            rule (str): The identifier of the decision rule.

        Returns:
            str: A descriptive string for the rule.
        """
        rule_descriptions = {
            "interest_rate": "Federal Funds Rate Adjustment",
            "inflation": "Composite CPI & PCE Inflation Adjustment",
            "yield_curve": "Yield Spread Adjustment",
            "employment": "Unemployment Rate Adjustment",
            "gdp": "GDP Growth Rate Adjustment",
            "inflation_expectations": "Breakeven Inflation Adjustment",
            "recession": "Recession Probability Adjustment",
            "credit_spread": "Credit Spread Adjustment",
            "allocation_adjustment": "Portfolio Allocation Diversification",
        }
        return rule_descriptions.get(
            rule, f"{rule.replace('_', ' ').title()} Adjustment"
        )

    def _add_rule_message(self, rule: str, message: str) -> None:
        """
        Add a message to the rule_messages dictionary for a given rule.

        Args:
            rule (str): The identifier of the decision rule.
            message (str): The message to associate with the rule.
        """
        if rule not in self.rule_messages:
            self.rule_messages[rule] = []
        self.rule_messages[rule].append(message)
        self.logger.debug(f"Added message to rule '{rule}': {message}")

    # Helper Methods

    def is_empty(self) -> bool:
        """
        Check if the portfolio has any allocations.

        Returns:
            bool: True if the portfolio has no allocations, False otherwise.
        """
        empty = not bool(self.portfolio.allocations)
        self.logger.debug(f"Portfolio is {'empty' if empty else 'not empty'}.")
        return empty

    def _validate_indicator_data(
        self, indicator_name: str, *keys: str
    ) -> Optional[Any]:
        """
        Safely retrieve nested data from analysis_results for a given indicator.

        Args:
            indicator_name (str): The name of the economic indicator.
            *keys (str): A sequence of keys to traverse the nested dictionary.

        Returns:
            Optional[Any]: The retrieved data if all keys are valid and not NaN, else None.

        Example:
            value = self._validate_indicator_data('GDP', 'growth_rate', 'annual')
        """
        analysis = self.analysis_results.get(indicator_name)
        if not analysis:
            self.logger.warning(
                f"No analysis data available for indicator '{indicator_name}'."
            )
            return None

        data: Any = analysis  # Allow data to be any type during traversal

        for key in keys:
            if isinstance(data, dict):
                data = data.get(key)
                if data is None:
                    self.logger.warning(
                        f"Data retrieval failed: '{key}' not available for indicator '{indicator_name}'."
                    )
                    return None
                if isinstance(data, float) and math.isnan(data):
                    self.logger.warning(
                        f"Data retrieval failed: '{key}' is NaN for indicator '{indicator_name}'."
                    )
                    return None
            else:
                self.logger.warning(
                    f"Data retrieval failed: '{key}' is not a dictionary for indicator '{indicator_name}'."
                )
                return None

        self.logger.debug(
            f"Successfully retrieved data for indicator '{indicator_name}': {data}"
        )
        return data
