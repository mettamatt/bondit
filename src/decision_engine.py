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
import shutil
import subprocess
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
        max_adjustment (float): Maximum percentage adjustment allowed for any operation.
        logger (logging.Logger): Logger instance for logging activities and debugging.
        baseline_allocations (Dict[str, float]): Strategic baseline allocations for portfolio assets.
        allocation_constraints (Dict[str, Dict[str, float]]): Allocation constraints for each asset.
        asset_risk_levels (Dict[str, float]): Risk levels assigned to each asset class.
        max_portfolio_risk (float): Maximum acceptable average risk level for the portfolio.
        analysis_results (Dict[str, Dict[str, Any]]): Results from analyzing each economic indicator.
    """

    def __init__(
        self,
        indicators: Dict[str, EconomicIndicator],
        portfolio: Portfolio,
        allocation_constraints: Dict[str, Dict[str, float]],
        logger: Optional[logging.Logger] = None,
    ) -> None:
        """
        Initialize the DecisionEngine with economic indicators and the investment portfolio.

        Args:
            indicators (Dict[str, EconomicIndicator]):
                Dictionary of EconomicIndicator objects used for portfolio analysis.
            portfolio (Portfolio):
                The Portfolio object to adjust based on economic indicators.
            allocation_constraints (Dict[str, Dict[str, float]]):
                Allocation constraints for each asset, defining minimum and maximum percentages.
            logger (Optional[logging.Logger], optional):
                Logger instance for logging messages. If None, a default logger is used.

        Raises:
            ValueError: If baseline allocations do not sum to 100% or violate allocation constraints.
        """
        self.indicators: Dict[str, EconomicIndicator] = indicators
        self.portfolio: Portfolio = portfolio
        self.max_adjustment: float = (
            25.0  # Maximum adjustment percentage for all operations
        )
        self.logger: logging.Logger = logger or logging.getLogger(
            "Bondit.DecisionEngine"
        )

        # Define strategic baseline allocations (total should sum to 100%)
        self.baseline_allocations: Dict[str, float] = {
            "VBIRX": 40.0,  # Vanguard Short-Term Bond Index Fund
            "VBLAX": 40.0,  # Vanguard Long-Term Bond Index Fund
            "VTAPX": 20.0,  # Vanguard Short-Term Inflation-Protected Securities Index Fund
        }

        # Define allocation constraints for each asset from centralized config
        self.allocation_constraints: Dict[str, Dict[str, float]] = (
            allocation_constraints
        )

        if not math.isclose(
            sum(self.baseline_allocations.values()), 100.0, abs_tol=0.1
        ):
            self.logger.critical(
                "Baseline allocations do not sum to 100%. Initialization aborted."
            )
            raise ValueError("Baseline allocations must sum to 100%.")

        # Validate baseline allocations against allocation constraints
        self._validate_allocation_constraints(self.baseline_allocations)

        # Define asset risk levels (higher number indicates higher risk)
        self.asset_risk_levels: Dict[str, float] = {
            "VBIRX": 1.0,  # Lower risk due to short duration
            "VBLAX": 2.5,  # Higher risk due to long duration
            "VTAPX": 1.5,  # Moderate risk with inflation protection
        }

        # Set maximum acceptable average portfolio risk level
        self.max_portfolio_risk: float = 2.0

        # Initialize analysis results
        self.analysis_results: Dict[str, Dict[str, Any]] = {}
        self.logger.debug("Initializing DecisionEngine.")
        for indicator_key, indicator in self.indicators.items():
            self.logger.debug(f"Analyzing indicator '{indicator_key}'.")
            self.analysis_results[indicator_key] = indicator.analyze_indicator()
        self.logger.info("DecisionEngine initialized with analysis results.")

    def apply_decision_rules(self) -> None:
        """
        Apply all decision rules to adjust the investment portfolio based on economic indicators.
        """
        self.logger.debug("Applying decision rules to the portfolio.")
        DecisionRules.apply_all_rules(self)
        self.portfolio.rebalance()
        self.logger.info(
            "Decision rules applied and allocations normalized successfully."
        )

    def _validate_allocation_constraints(self, allocations: Dict[str, float]) -> None:
        """
        Validate that the provided allocations adhere to the defined allocation constraints.

        Args:
            allocations (Dict[str, float]): The allocations to validate.

        Raises:
            ValueError: If any allocation violates its constraints.
        """
        for asset, allocation in allocations.items():
            constraints = self.allocation_constraints.get(asset)
            if constraints:
                min_alloc = constraints["min"]
                max_alloc = constraints["max"]
                if allocation < min_alloc or allocation > max_alloc:
                    self.logger.critical(
                        f"Initial allocation for '{asset}' is {allocation}%, which "
                        f"violates constraints ({min_alloc}%-{max_alloc}%). Initialization aborted."
                    )
                    raise ValueError(
                        f"Initial allocation for '{asset}' must be between {min_alloc}% and {max_alloc}%."
                    )
        self.logger.debug("Initial allocations comply with allocation constraints.")

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
        self, asset: str, amount: float, rule: str, rule_weight: float, rationale: str
    ) -> None:
        """
        Adjust the allocation for a specific asset based on a decision rule, enforcing allocation constraints.

        Args:
            asset (str): The asset to adjust.
            amount (float): The adjustment amount (positive to increase, negative to decrease).
            rule (str): The decision rule that proposes this adjustment.
            rule_weight (float): The weight of the decision rule.
            rationale (str): The strategic justification for the action.
        """
        current_allocation = self.portfolio.allocations.get(asset, 0.0)
        constraints = self.allocation_constraints.get(asset, {"min": 0.0, "max": 100.0})
        min_alloc = constraints["min"]
        max_alloc = constraints["max"]

        proposed_allocation = current_allocation + amount

        # Enforce allocation constraints
        if proposed_allocation < min_alloc:
            adjusted_amount = min_alloc - current_allocation
            self.logger.warning(
                f"Adjustment for '{asset}' by {amount:.2f}% would fall below the minimum of {min_alloc}%. "
                f"Capping adjustment to {adjusted_amount:.2f}%."
            )
            amount = adjusted_amount
        elif proposed_allocation > max_alloc:
            adjusted_amount = max_alloc - current_allocation
            self.logger.warning(
                f"Adjustment for '{asset}' by {amount:.2f}% would exceed the maximum of {max_alloc}%. "
                f"Capping adjustment to {adjusted_amount:.2f}%."
            )
            amount = adjusted_amount

        if amount == 0.0:
            self.logger.info(
                f"No adjustment made for '{asset}' as it is already at its allocation constraint."
            )
            return  # No adjustment needed

        # Apply the adjustment to the portfolio
        try:
            self.portfolio.adjust_allocation(
                asset, amount, rule, rule_weight, rationale
            )
            self.logger.info(
                f"Adjusted '{asset}' by {amount:.2f}% based on rule '{rule}': {rationale}"
            )
        except Exception as e:
            self.logger.error(
                f"Failed to adjust allocation for '{asset}' by {amount:.2f}%: {e}"
            )

    def generate_rebalancing_report(self) -> str:
        """
        Generate a Markdown-formatted rebalancing report based on current economic indicators,
        portfolio adjustments, and updated portfolio allocations.

        Returns:
            str: A string containing the formatted Markdown report.
        """
        report_lines = []
        report_date = datetime.datetime.now().strftime("%Y-%m-%d")

        # Report Header
        report_lines.append(f"## Rebalancing Report Date: {report_date}\n\n")

        # Updated Portfolio Allocations Section
        report_lines.append("### **Updated Portfolio Allocations**\n")
        report_lines.append("| **Asset**             | **Allocation** |")
        report_lines.append("|-----------------------|----------------|")
        for asset, allocation in self.portfolio.get_allocations().items():
            report_lines.append(f"| {asset:<21} | {allocation:.2f}%        |")
        report_lines.append("\n---\n")

        # Economic Indicators Overview Section
        report_lines.append("### **Economic Indicators Overview**\n")
        report_lines.append(
            "| **Economic Indicator**       | **Overall Trend** | **1-Year Signal** |"
        )
        report_lines.append(
            "|------------------------------|--------------------|--------------------|"
        )

        for indicator_key, indicator in self.indicators.items():
            analysis = self.analysis_results.get(indicator_key, {})
            overall_trend = analysis.get("overall_trend", "N/A")
            one_year_signal = analysis.get("1y", {}).get("signal", "N/A")
            display_name = (
                indicator.config.name
            )  # Use the 'name' attribute from IndicatorConfig
            report_lines.append(
                f"| {display_name:<30} | {overall_trend:<18} | {one_year_signal:<18} |"
            )
        report_lines.append("\n---\n")

        # Rebalancing Scenario Section
        report_lines.append("### **Rebalancing Scenario**\n")
        report_lines.append("#### **Portfolio Adjustments**\n")
        report_lines.append(
            "| **Adjustment Type**                | **Asset**    | **Action**                                     | **Amount** | **Rationale**                                                                                         |"
        )
        report_lines.append(
            "|------------------------------------|--------------|------------------------------------------------|------------|-------------------------------------------------------------------------------------------------------|"
        )

        # Retrieve adjustments from the Portfolio
        for adjustment in self.portfolio.get_adjustments():
            report_lines.append(
                f"| **{adjustment['Adjustment Type']}**  | {adjustment['Asset']} | {adjustment['Action']} | {adjustment['Amount']} | {adjustment['Rationale']} |"
            )

        report_lines.append("\n---\n")

        # Notes Section
        report_lines.append("### **Notes**\n")
        report_lines.append(
            "- **Overall Trend:** Represents the most frequently occurring trend signal derived from the analysis of all economic indicators."
        )
        report_lines.append(
            "- **1-Year Trend Signal:** Reflects the trend signal specifically for each economic indicator over the past year.\n"
        )
        report_lines.append("\n---\n")

        # Report Footer
        report_lines.append("**Report Generated by:** Bondit v1.0 \n---")

        # Combine all lines into a single string
        return "\n".join(report_lines)

    def save_rebalancing_report(
        self, file_path: str = "rebalancing_report.md", view: bool = False
    ) -> None:
        """
        Save the generated rebalancing report to a markdown file and optionally view it using mdv.

        Args:
            file_path (str, optional): The path to the file where the report will be saved.
                                       Defaults to "rebalancing_report.md".
            view (bool, optional): Whether to open the report in Terminal Markdown Viewer after saving.
                                   Defaults to False.

        Raises:
            IOError: If the file cannot be written.
        """
        report = self.generate_rebalancing_report()
        try:
            with open(file_path, "w") as file:
                file.write(report)
            self.logger.info(f"Rebalancing report saved to {file_path}.")
        except IOError as e:
            self.logger.error(f"Failed to save rebalancing report to {file_path}: {e}")
            raise

        if view:
            self.view_rebalancing_report(file_path)

    def view_rebalancing_report(self, file_path: str) -> None:
        """
        Open the saved rebalancing report using Terminal Markdown Viewer (mdv).

        Args:
            file_path (str): The path to the markdown file to be viewed.
        """
        # Check if mdv is installed
        if shutil.which("mdv") is None:
            self.logger.error(
                "Terminal Markdown Viewer (mdv) is not installed or not found in PATH.\n"
                "Please install it by running 'pip install mdv' and ensure it's accessible from your terminal."
            )
            return

        # Attempt to open the report with mdv
        try:
            subprocess.run(["mdv", file_path], check=True)
            self.logger.info(f"Opened '{file_path}' in Terminal Markdown Viewer (mdv).")
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to open '{file_path}' with mdv: {e}")
        except Exception as e:
            self.logger.error(
                f"An unexpected error occurred while opening the report: {e}"
            )

    # Helper Methods

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
