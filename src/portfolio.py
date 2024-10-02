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

import datetime
import logging
import math
from typing import Any, Dict, List, Optional


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
        adjustment_history (List[Dict[str, Any]]): Records of all adjustments made to the portfolio.
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
                Minimum allocation percentages for each asset type.
            max_allocations (Optional[Dict[str, float]]):
                Maximum allocation percentages for each asset type.
            logger (Optional[logging.Logger], optional):
                Logger instance for logging messages. If None, a default logger is used.

        Raises:
            ValueError: If initial allocations do not sum to 100%.
        """
        self.logger: logging.Logger = logger or logging.getLogger("Bondit.Portfolio")

        # Initialize adjustment history
        self.adjustment_history: List[Dict[str, Any]] = []

        # Set default allocations if none provided
        default_allocations = {
            "VBIRX": 40.0,  # Vanguard Short-Term Bond Index Fund
            "VBLAX": 40.0,  # Vanguard Long-Term Bond Index Fund
            "VTAPX": 20.0,  # Vanguard Short-Term Inflation-Protected Securities Index Fund
        }

        self.allocations: Dict[str, float] = (
            allocations.copy() if allocations else default_allocations.copy()
        )

        # Assign min and max allocations from parameters
        self.min_allocations: Dict[str, float] = (
            min_allocations.copy() if min_allocations else {}
        )
        self.max_allocations: Dict[str, float] = (
            max_allocations.copy() if max_allocations else {}
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
        self,
        asset_type: str,
        amount: float,
        rule_key: str,
        rule_weight: float,
        rationale: str = "",
    ) -> None:
        """
        Adjust the allocation for a specific asset type without enforcing constraints immediately.

        Args:
            asset_type (str): The name of the asset type.
            amount (float): The amount to adjust (positive or negative).
            rule_key (str): The key of the rule making the adjustment.
            rule_weight (float): The weight of the rule influencing the adjustment.
            rationale (str): The rationale for the adjustment.
        """
        self.logger.debug(f"Attempting to adjust '{asset_type}' by {amount:+.2f}%.")

        old_alloc = self.allocations.get(asset_type, 0.0)
        proposed_alloc = old_alloc + amount

        # Record the adjustment without enforcing constraints
        self.allocations[asset_type] = proposed_alloc

        self.logger.info(
            f"Adjustment by Rule '{rule_key}' (Weight: {rule_weight}) - "
            f"Asset: '{asset_type}', "
            f"Amount: {amount:+.2f}%, "
            f"Allocation: {old_alloc:.2f}% -> {self.allocations[asset_type]:.2f}%. "
            f"Rationale: {rationale}"
        )

        # Record the adjustment
        adjustment_record = {
            "Adjustment Type": rule_key,
            "Asset": asset_type,
            "Action": f"{'Increased' if amount > 0 else 'Decreased'} by {abs(amount):.2f}%.",
            "Amount": f"{amount:+.2f}%",
            "Rationale": rationale,
            "Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        self.adjustment_history.append(adjustment_record)

        # Constraints will be enforced during rebalance

    def get_adjustments(self) -> List[Dict[str, Any]]:
        """
        Retrieve the adjustment history of the portfolio.

        Returns:
            List[Dict[str, Any]]: A list of adjustment records.
        """
        self.logger.debug("Retrieving adjustment history.")
        return self.adjustment_history.copy()

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

        # Validate allocations after setting
        self.validate_allocations()

    def rebalance(self) -> None:
        """
        Rebalance the portfolio to ensure total allocations sum to 100%,
        redistributing excess or deficit among assets within their constraints.
        """
        self.logger.debug("Starting rebalance process.")

        # Enforce min and max constraints
        for asset in self.allocations:
            alloc = self.allocations[asset]
            min_alloc = self.min_allocations.get(asset, 0.0)
            max_alloc = self.max_allocations.get(asset, 100.0)

            if alloc < min_alloc:
                self.logger.debug(
                    f"Adjusting '{asset}' up to its minimum allocation of {min_alloc}%."
                )
                self.allocations[asset] = min_alloc
            elif alloc > max_alloc:
                self.logger.debug(
                    f"Adjusting '{asset}' down to its maximum allocation of {max_alloc}%."
                )
                self.allocations[asset] = max_alloc

        # Iteratively adjust allocations
        max_iterations = 20
        iteration = 0

        while iteration < max_iterations:
            total_alloc = sum(self.allocations.values())
            deviation = total_alloc - 100.0

            self.logger.debug(
                f"Iteration {iteration}: Total allocations sum to {total_alloc:.2f}%, deviation of {deviation:+.2f}%."
            )

            if abs(deviation) < 1e-2:
                self.logger.debug("Allocations successfully normalized to sum to 100%.")
                break

            # Identify adjustable assets
            if deviation > 0:
                # Need to decrease allocations
                adjustable_assets = [
                    asset
                    for asset in self.allocations
                    if self.allocations[asset] > self.min_allocations.get(asset, 0.0)
                ]
            else:
                # Need to increase allocations
                adjustable_assets = [
                    asset
                    for asset in self.allocations
                    if self.allocations[asset] < self.max_allocations.get(asset, 100.0)
                ]

            if not adjustable_assets:
                self.logger.warning(
                    "No adjustable assets available to rebalance allocations."
                )
                break

            # Calculate total flexibility
            if deviation > 0:
                total_flex = sum(
                    self.allocations[asset] - self.min_allocations.get(asset, 0.0)
                    for asset in adjustable_assets
                )
            else:
                total_flex = sum(
                    self.max_allocations.get(asset, 100.0) - self.allocations[asset]
                    for asset in adjustable_assets
                )

            self.logger.debug(
                f"Total flexibility available for adjustment: {total_flex:.2f}%."
            )

            if total_flex == 0:
                self.logger.warning(
                    "Total flexible allocation is zero. Cannot rebalance further."
                )
                break

            # Adjust allocations proportionally
            for asset in adjustable_assets:
                if deviation > 0:
                    # Decrease allocation
                    flex = self.allocations[asset] - self.min_allocations.get(
                        asset, 0.0
                    )
                    if flex <= 0:
                        self.logger.debug(f"No flexibility to decrease '{asset}'.")
                        continue
                    adjustment = (flex / total_flex) * deviation
                    adjustment = min(
                        adjustment, flex
                    )  # Ensure we don't go below min_alloc
                    self.allocations[asset] -= adjustment
                    self.logger.debug(
                        f"Decreased '{asset}' by {adjustment:.2f}% to {self.allocations[asset]:.2f}%."
                    )
                else:
                    # Increase allocation
                    flex = (
                        self.max_allocations.get(asset, 100.0) - self.allocations[asset]
                    )
                    if flex <= 0:
                        self.logger.debug(f"No flexibility to increase '{asset}'.")
                        continue
                    adjustment = (flex / total_flex) * (-deviation)
                    adjustment = min(
                        adjustment, flex
                    )  # Ensure we don't exceed max_alloc
                    self.allocations[asset] += adjustment
                    self.logger.debug(
                        f"Increased '{asset}' by {adjustment:.2f}% to {self.allocations[asset]:.2f}%."
                    )

            iteration += 1

        # Final validation
        total_alloc = sum(self.allocations.values())
        if not math.isclose(total_alloc, 100.0, abs_tol=1e-2):
            self.logger.warning(
                f"Total allocations after rebalance sum to {total_alloc:.2f}%, which deviates from the expected 100%."
            )
        else:
            self.logger.debug("Allocations successfully rebalanced to sum to 100%.")
        self.validate_allocations()
        self.logger.info("Rebalance complete.")

    def get_allocations(self) -> Dict[str, float]:
        """
        Retrieve the current allocations of the portfolio.

        Returns:
            Dict[str, float]: A copy of the current allocation percentages for each asset type.
        """
        return self.allocations.copy()

    def validate_allocations(self) -> None:
        """
        Validate that all allocations adhere to their respective constraints and that the total
        allocations sum to approximately 100%. Logs detailed warnings for any violations.
        """
        self.logger.debug("Validating portfolio allocations.")
        total_alloc = sum(self.allocations.values())

        # Check each asset's allocation against its constraints
        for asset, alloc in self.allocations.items():
            min_alloc = self.min_allocations.get(asset, 0.0)
            max_alloc = self.max_allocations.get(asset, 100.0)
            if alloc < min_alloc:
                self.logger.warning(
                    f"Allocation for '{asset}' is below the minimum of {min_alloc}%. Current allocation: {alloc:.2f}%."
                )
            if alloc > max_alloc:
                self.logger.warning(
                    f"Allocation for '{asset}' exceeds the maximum of {max_alloc}%. Current allocation: {alloc:.2f}%."
                )

        # Check if total allocations sum to 100%
        if not math.isclose(total_alloc, 100.0, abs_tol=0.1):
            self.logger.warning(
                f"Total allocations sum to {total_alloc:.2f}%, which deviates from the expected 100%."
            )
        else:
            self.logger.debug(
                "All allocations are within defined constraints and sum to 100%."
            )

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
