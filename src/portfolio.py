# src/portfolio.py

"""
portfolio.py

This module defines the `Portfolio` class, which manages investment allocations across various
asset types. It provides functionalities to set and adjust allocations, enforce allocation
constraints, and ensure the portfolio remains balanced and aligned with strategic objectives.

Classes:
    Portfolio: Manages investment allocations, ensuring they adhere to predefined constraints
               and maintain a balanced portfolio.
"""

import logging
import math
from typing import Dict, Optional


class Portfolio:
    """
    Manages investment allocations across different asset types.

    The `Portfolio` class allows for setting and adjusting allocations while enforcing minimum
    and maximum constraints for each asset. It ensures that the total allocations sum to 100%
    and maintains the portfolio's integrity through rebalancing.

    Attributes:
        allocations (Dict[str, float]): Current allocation percentages for each asset type.
        min_allocations (Dict[str, float]): Minimum allocation percentages for each asset type.
        max_allocations (Dict[str, float]): Maximum allocation percentages for each asset type.
        logger (logging.Logger): Logger instance for logging activities and debugging.
    """

    def __init__(
        self,
        allocations: Optional[Dict[str, float]] = None,
        min_allocations: Optional[Dict[str, float]] = None,
        max_allocations: Optional[Dict[str, float]] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        """
        Initialize the Portfolio with optional allocations and constraints.

        This constructor sets up the portfolio's allocations and ensures that they adhere
        to the defined minimum and maximum constraints. It also validates that the initial
        allocations sum to 100%, maintaining portfolio integrity from the outset.

        Args:
            allocations (Optional[Dict[str, float]]):
                Initial allocation percentages for each asset type. If None, starts with a default portfolio.
            min_allocations (Optional[Dict[str, float]]):
                Minimum allocation percentages for each asset type. Defaults are set to 0% for all assets.
            max_allocations (Optional[Dict[str, float]]):
                Maximum allocation percentages for each asset type, based on provided constraints.
            logger (Optional[logging.Logger], optional):
                Logger instance for logging messages. If None, a default logger is used.

        Raises:
            ValueError: If initial allocations do not sum to 100%.

        Example:
            >>> portfolio = Portfolio(
            ...     allocations={"Long-Term Government Bond": 25.0, "Long-Term Investment-Grade Corporate Bond": 25.0, ...},
            ...     min_allocations={"Long-Term Government Bond": 0.0, "Long-Term Investment-Grade Corporate Bond": 0.0, ...},
            ...     max_allocations={"Long-Term Government Bond": 30.0, "Long-Term Investment-Grade Corporate Bond": 25.0, ...}
            ... )
        """
        self.logger: logging.Logger = logger or logging.getLogger("Bondit.Portfolio")
        
        # Set default allocations if none provided
        default_allocations = {
            "Long-Term Government Bond": 25.0,
            "Long-Term Investment-Grade Corporate Bond": 20.0,
            "Intermediate-Term Government Bond": 15.0,
            "Intermediate-Term Investment-Grade Corporate Bond": 15.0,
            "Short-Term Government Bond": 5.0,
            "Short-Term Investment-Grade Corporate Bond": 5.0,
            "TIPS": 10.0,
            "Intermediate-Term National Municipal Bond": 5.0,
            "Intermediate-Term State Municipal Bond": 5.0,
            "Short-Term National Municipal Bond": 2.5,
            "Short-Term State Municipal Bond": 2.5,
            "International Bond": 10.0,
        }
        
        self.allocations: Dict[str, float] = allocations.copy() if allocations else default_allocations.copy()
        
        # Set minimum allocations to 0% for all assets
        default_min_allocations = {asset: 0.0 for asset in self.allocations.keys()}
        self.min_allocations: Dict[str, float] = (
            min_allocations.copy() if min_allocations else default_min_allocations
        )
        
        # Set maximum allocations based on the provided constraints
        default_max_allocations = {
            "Long-Term Government Bond": 30.0,
            "Long-Term Investment-Grade Corporate Bond": 25.0,
            "Intermediate-Term Government Bond": 20.0,
            "Intermediate-Term Investment-Grade Corporate Bond": 20.0,
            "Short-Term Government Bond": 10.0,
            "Short-Term Investment-Grade Corporate Bond": 10.0,
            "TIPS": 15.0,
            "Intermediate-Term National Municipal Bond": 10.0,
            "Intermediate-Term State Municipal Bond": 10.0,
            "Short-Term National Municipal Bond": 5.0,
            "Short-Term State Municipal Bond": 5.0,
            "International Bond": 15.0,
        }
        self.max_allocations: Dict[str, float] = (
            max_allocations.copy() if max_allocations else default_max_allocations
        )

        self.logger.debug(
            f"Initializing Portfolio with allocations: {self.allocations}"
        )
        self.logger.debug(f"Minimum allocations: {self.min_allocations}")
        self.logger.debug(f"Maximum allocations: {self.max_allocations}")

        # Ensure that max_allocations are set for all assets in allocations
        for asset in self.allocations:
            if asset not in self.max_allocations:
                self.max_allocations[asset] = 100.0  # Default max allocation
                self.logger.debug(f"Set default max allocation for '{asset}': 100.0%")
            if asset not in self.min_allocations:
                self.min_allocations[asset] = 0.0  # Default min allocation

        # Validate that allocations adhere to min and max constraints
        for asset in self.allocations:
            alloc = self.allocations[asset]
            min_alloc = self.min_allocations.get(asset, 0.0)
            max_alloc = self.max_allocations.get(asset, 100.0)
            if alloc < min_alloc:
                self.logger.warning(
                    f"Allocation for '{asset}' is below the minimum of {min_alloc}%. Adjusting to minimum."
                )
                self.allocations[asset] = min_alloc
            elif alloc > max_alloc:
                self.logger.warning(
                    f"Allocation for '{asset}' exceeds the maximum of {max_alloc}%. Adjusting to maximum."
                )
                self.allocations[asset] = max_alloc

        total_alloc = sum(self.allocations.values())
        self.logger.debug(f"Total initial allocation: {total_alloc}%")

        # Use math.isclose to handle floating-point precision
        if not math.isclose(total_alloc, 100.0, abs_tol=1e-2):
            self.logger.error(
                f"Initial allocations sum to {total_alloc}%, expected 100%."
            )
            raise ValueError("Initial allocations must sum to 100%.")
        self.logger.info(
            "Portfolio initialized successfully with allocations summing to 100%."
        )

    def adjust_allocation(
        self, asset_type: str, amount: float, rule_key: str, rule_weight: float, rationale: str = ""
    ) -> None:
        """
        Adjust the allocation for a specific asset type.

        Args:
            asset_type (str): The name of the asset type.
            amount (float): The amount to adjust (positive or negative).
            rule_key (str): The key of the rule making the adjustment.
            rule_weight (float): The weight of the rule influencing the adjustment.
            rationale (str): The rationale for the adjustment.
        """
        old_alloc = self.allocations.get(asset_type, 0.0)
        proposed_alloc = old_alloc + amount

        # Enforce min and max constraints
        min_alloc = self.min_allocations.get(asset_type, 0.0)
        max_alloc = self.max_allocations.get(asset_type, 100.0)
        adjusted_amount = amount  # To track the final adjustment after constraints

        if proposed_alloc < min_alloc:
            adjusted_amount = min_alloc - old_alloc
            self.logger.warning(
                f"Proposed allocation for '{asset_type}' is {proposed_alloc:.2f}%, "
                f"which is below the minimum of {min_alloc}%. "
                f"Adjusting allocation from {proposed_alloc:.2f}% to {min_alloc}%."
            )
            self.allocations[asset_type] = min_alloc
        elif proposed_alloc > max_alloc:
            adjusted_amount = max_alloc - old_alloc
            self.logger.warning(
                f"Proposed allocation for '{asset_type}' is {proposed_alloc:.2f}%, "
                f"which exceeds the maximum of {max_alloc}%. "
                f"Adjusting allocation from {proposed_alloc:.2f}% to {max_alloc}%."
            )
            self.allocations[asset_type] = max_alloc
        else:
            self.allocations[asset_type] = proposed_alloc

        # Log the adjustment with detailed information
        self.logger.info(
            f"Adjustment made by Rule '{rule_key}' (Weight: {rule_weight}) - "
            f"Asset: '{asset_type}', "
            f"Amount: {adjusted_amount:+.2f}%, "
            f"Allocation: {old_alloc:.2f}% -> {self.allocations[asset_type]:.2f}%. "
            f"Rationale: {rationale}"
        )

    def set_allocations(self, allocations: Dict[str, float]) -> None:
        """
        Set the allocations for multiple assets at once, enforcing constraints.

        Args:
            allocations (Dict[str, float]): A dictionary of asset allocations to set.

        Raises:
            ValueError: If the provided allocations do not sum to approximately 100%.
        """
        self.logger.debug(f"Setting allocations: {allocations}")
        total_alloc = sum(allocations.values())

        if not math.isclose(total_alloc, 100.0, abs_tol=0.1):
            self.logger.error(
                f"Provided allocations sum to {total_alloc}%, expected approximately 100%."
            )
            raise ValueError("Provided allocations must sum to approximately 100%.")

        for asset, alloc in allocations.items():
            self.set_allocation(asset, alloc)

        self.logger.info("Allocations set successfully.")

    def set_allocation(self, asset_type: str, alloc: float) -> None:
        """
        Set the allocation for a specific asset type, enforcing constraints.

        This method directly sets the allocation percentage for a given asset type,
        ensuring that the new allocation adheres to the predefined minimum and maximum
        constraints.

        Args:
            asset_type (str): The asset type to set (e.g., "TIPS").
            alloc (float): The allocation percentage to set.

        Example:
            >>> portfolio.set_allocation("TIPS", 5.0)
            Sets "TIPS" allocation to 5.0%.
        """
        self.logger.debug(f"Setting allocation for '{asset_type}' to {alloc}%.")

        # Enforce constraints
        min_alloc = self.min_allocations.get(asset_type, 0.0)
        max_alloc = self.max_allocations.get(asset_type, 100.0)

        constrained_alloc = max(min_alloc, min(alloc, max_alloc))
        self.allocations[asset_type] = constrained_alloc

        if constrained_alloc != alloc:
            self.logger.debug(
                f"Allocation for '{asset_type}' adjusted to constraints: {constrained_alloc:.2f}% (requested {alloc:.2f}%)."
            )

        # If asset is new, initialize its allocation
        if asset_type not in self.allocations:
            self.logger.info(
                f"Adding new asset '{asset_type}' to portfolio with allocation {constrained_alloc:.2f}%."
            )

        self.logger.info(
            f"Set allocation for '{asset_type}': {constrained_alloc:.2f}%."
        )

    def rebalance(self) -> None:
        """
        Rebalance the portfolio to ensure total allocations sum to 100%.

        This method adjusts all asset allocations proportionally to correct any deviations
        from the total allocation target of 100%. It ensures that the portfolio remains balanced
        and that allocations adhere to strategic objectives.

        Example:
            >>> portfolio.rebalance()
            Rebalances the portfolio allocations to sum to 100%.
        """
        total = sum(self.allocations.values())
        self.logger.debug(f"Total allocations before rebalancing: {total}%")

        # Use math.isclose to handle floating-point precision
        if not math.isclose(total, 100.0, abs_tol=0.1) and total != 0.0:
            factor = 100.0 / total
            self.logger.debug(f"Rebalancing factor: {factor}")

            for asset_type in self.allocations:
                old_alloc = self.allocations[asset_type]
                new_alloc = old_alloc * factor
                # Enforce max allocations after rebalancing
                max_alloc = self.max_allocations.get(asset_type, 100.0)
                new_alloc = min(new_alloc, max_alloc)
                self.allocations[asset_type] = new_alloc
                self.logger.debug(
                    f"Rebalanced '{asset_type}': {old_alloc}% -> {new_alloc:.2f}%"
                )

            self.logger.info("Portfolio rebalanced to sum to 100%.")
        else:
            if math.isclose(total, 100.0, abs_tol=0.1):
                self.logger.debug(
                    "No rebalancing needed. Allocations already sum to 100%."
                )
            else:
                self.logger.warning("Total allocations are 0%. Rebalancing skipped.")

    def get_allocations(self) -> Dict[str, float]:
        """
        Retrieve the current allocations of the portfolio.

        Returns:
            Dict[str, float]: A copy of the current allocation percentages for each asset type.

        Example:
            >>> current_allocations = portfolio.get_allocations()
            >>> print(current_allocations)
            {'Long-Term Government Bond': 30.0, 'Short-Term Bonds': 10.0}
        """
        self.logger.debug(f"Retrieving current allocations: {self.allocations}")
        return self.allocations.copy()

    def validate_total_allocation(self) -> None:
        """
        Validate that the total allocations sum to approximately 100%.

        Logs a warning if the total allocation deviates from 100%, aiding in maintaining portfolio integrity.

        Example:
            >>> portfolio.validate_total_allocation()
            Validates that allocations sum to approximately 100%.
        """
        total_alloc = sum(self.allocations.values())
        if not math.isclose(total_alloc, 100.0, abs_tol=0.1):
            self.logger.warning(
                f"Total allocations sum to {total_alloc:.2f}%, which deviates from the expected 100%."
            )
        else:
            self.logger.debug("Total allocations correctly sum to 100%.")

    def get_all_asset_types(self) -> list:
        """
        Retrieve a list of all asset types in the portfolio.

        Returns:
            list: A list containing the names of all asset types in the portfolio.

        Example:
            >>> asset_types = portfolio.get_all_asset_types()
            >>> print(asset_types)
            ['Long-Term Government Bond', 'Short-Term Bonds']
        """
        self.logger.debug("Retrieving all asset types in the portfolio.")
        return list(self.allocations.keys())
