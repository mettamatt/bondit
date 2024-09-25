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
from typing import Any, Dict, List, Optional

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
        adjustments (List[Dict[str, Any]]): List of individual asset adjustments made.
        max_adjustment (float): Maximum percentage adjustment allowed for any operation.
        logger (logging.Logger): Logger instance for logging activities and debugging.
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
        self.rule_messages: Dict[str, List[str]] = {}
        self.adjustments: List[Dict[str, Any]] = []
        self.max_adjustment: float = (
            10.0  # Maximum adjustment percentage for all operations
        )
        self.logger: logging.Logger = logger or logging.getLogger(
            "Bondit.DecisionEngine"
        )

        # Define strategic baseline allocations (total should sum to 100%)
        self.baseline_allocations: Dict[str, float] = {
            "Intermediate-Term Government Bond": 15.0,
            "Intermediate-Term Investment-Grade Corporate Bond": 10.0,
            "Intermediate-Term National Municipal Bond": 10.0,
            "Intermediate-Term State Municipal Bond": 5.0,
            "Long-Term Government Bond": 10.0,
            "Long-Term Investment-Grade Corporate Bond": 10.0,
            "Short-Term Government Bond": 10.0,
            "Short-Term Investment-Grade Corporate Bond": 5.0,
            "Short-Term National Municipal Bond": 5.0,
            "International Bond": 10.0,
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
            "Intermediate-Term Government Bond": 2.0,
            "Intermediate-Term Investment-Grade Corporate Bond": 3.0,
            "Intermediate-Term National Municipal Bond": 2.5,
            "Intermediate-Term State Municipal Bond": 2.5,
            "Long-Term Government Bond": 2.5,
            "Long-Term Investment-Grade Corporate Bond": 3.5,
            "Short-Term Government Bond": 1.0,
            "Short-Term Investment-Grade Corporate Bond": 1.5,
            "Short-Term National Municipal Bond": 1.5,
            "International Bond": 3.5,
        }

        # Set maximum acceptable average portfolio risk level
        self.max_portfolio_risk: float = 2.5

        # Initialize analysis results
        self.analysis_results: Dict[str, Dict[str, Any]] = {}
        self.logger.debug("Initializing DecisionEngine.")
        for indicator_key, indicator in self.indicators.items():
            self.logger.debug(f"Analyzing indicator '{indicator_key}'.")
            self.analysis_results[indicator_key] = indicator.analyze_indicator()
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

        This method applies each decision rule in a prioritized order and adjusts the portfolio
        allocations accordingly. After adjustments, it generates a rebalancing report summarizing
        the actions taken.
        """
        self.logger.info("Applying decision rules.")
        self.rule_messages = {}
        self.adjustments = []  # Reset adjustments for the current cycle

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
        self, asset: str, amount: float, rule: str, rule_weight: float, rationale: str
    ) -> None:
        """
        Adjust the allocation for a specific asset based on a decision rule.

        Args:
            asset (str): The asset to adjust.
            amount (float): The adjustment amount (positive to increase, negative to decrease).
            rule (str): The decision rule that proposes this adjustment.
            rule_weight (float): The weight of the decision rule.
            rationale (str): The strategic justification for the action.
        """
        # Limit adjustment to the maximum allowed
        if abs(amount) > self.max_adjustment:
            adjusted_amount = math.copysign(self.max_adjustment, amount)
            self.logger.warning(
                f"Adjustment for '{asset}' by {amount:.2f}% exceeds max adjustment. Capped to {adjusted_amount:.2f}%."
            )
            amount = adjusted_amount

        self.portfolio.adjust_allocation(asset, amount, rule, rule_weight)

        adjustment_record = {
            "Adjustment Type": self.indicators[rule].config.description,
            "Asset": asset,
            "Action": f"{'Increased' if amount > 0 else 'Decreased'} by {abs(amount):.2f}%.",
            "Amount": f"{'+' if amount > 0 else ''}{amount:.2f}%",
            "Rationale": rationale,
        }
        self.adjustments.append(adjustment_record)

        self.logger.info(
            f"{adjustment_record['Adjustment Type']}: {adjustment_record['Action']} Rationale: {adjustment_record['Rationale']}"
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
        report_lines.append("| **Asset**                            | **Allocation** |")
        report_lines.append(
            "|--------------------------------------|-----------------|"
        )
        for asset, allocation in self.portfolio.get_allocations().items():
            report_lines.append(f"| {asset:<36} | {allocation:.2f}%          |")
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
            "| **Adjustment Type**                | **Asset**                      | **Action**                                     | **Amount** | **Rationale**                                                                                         |"
        )
        report_lines.append(
            "|------------------------------------|--------------------------------|------------------------------------------------|------------|-------------------------------------------------------------------------------------------------------|"
        )

        for adjustment in self.adjustments:
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
        except Exception as e:
            self.logger.error(f"Failed to save rebalancing report to {file_path}: {e}")
            return  # Early exit if saving fails

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
