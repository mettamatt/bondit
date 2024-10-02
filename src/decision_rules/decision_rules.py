# src/decision_rules/decision_rules.py

"""
This module defines individual decision rules for the DecisionEngine. Each rule analyzes
specific economic indicators and proposes adjustments to the investment portfolio accordingly.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..decision_engine import DecisionEngine


class DecisionRules:
    """
    Container for all decision rules applied by the DecisionEngine.

    Each static method within this class represents a distinct decision rule that evaluates
    specific economic indicators and proposes corresponding portfolio adjustments.
    """

    @staticmethod
    def apply_all_rules(engine: "DecisionEngine") -> None:
        """
        Apply all decision rules to the portfolio in a prioritized order.

        Args:
            engine (DecisionEngine): The instance of DecisionEngine to apply rules on.
        """
        logger = engine.logger

        # Prioritized list of rule methods to apply
        rules = [
            DecisionRules.apply_interest_rate_rule,  # High priority
            DecisionRules.apply_inflation_rule,  # High priority
            DecisionRules.apply_inflation_expectations_rule,  # High priority
            DecisionRules.apply_yield_curve_rule,  # Medium-High priority
            DecisionRules.apply_recession_probability_rule,  # Medium-High priority
            DecisionRules.apply_credit_spread_rule,  # Medium priority
            DecisionRules.apply_gdp_growth_rule,  # Medium priority
            DecisionRules.apply_employment_rule,  # Medium-Low priority
        ]

        for rule in rules:
            try:
                rule(engine)
            except Exception as e:
                logger.error(f"Error applying rule {rule.__name__}: {e}")

    @staticmethod
    def apply_interest_rate_rule(engine: "DecisionEngine") -> None:
        """
        Adjust portfolio allocations based on changes in the Federal Funds Rate.
        """
        indicator_key = "fed_funds"
        z_score = engine._validate_indicator_data(
            indicator_key, "statistics", "z_score"
        )
        if z_score is None:
            engine.logger.warning(
                f"Interest Rate Rule: Missing 'z_score' data for indicator '{indicator_key}'. Rule skipped."
            )
            return

        upper_threshold, lower_threshold = engine.indicators[
            indicator_key
        ].config.thresholds

        # Retrieve rule weight from IndicatorConfig
        rule_weight = engine.get_rule_weight(indicator_key)
        proposed_adjustment = min(engine.max_adjustment * rule_weight, abs(z_score))

        rule_name = "Interest Rate Adjustment"

        if z_score > upper_threshold:
            # Rates are significantly higher than usual; reduce duration
            rationale = "High interest rates can lead to falling bond prices; reducing duration mitigates risk."

            # Apply adjustments: reduce VBLAX, increase VBIRX
            assets_to_reduce = ["VBLAX"]
            assets_to_increase = ["VBIRX"]

            for asset in assets_to_reduce:
                engine.adjust_allocation(
                    asset,
                    -proposed_adjustment,
                    indicator_key,
                    rule_weight,
                    rationale,
                )

            for asset in assets_to_increase:
                engine.adjust_allocation(
                    asset,
                    proposed_adjustment,
                    indicator_key,
                    rule_weight,
                    rationale,
                )

            action = f"Reduced allocation to VBLAX and increased allocation to VBIRX by {proposed_adjustment:.2f}%."
            engine.logger.info(f"{rule_name}: {action} Rationale: {rationale}")

        elif z_score < lower_threshold:
            # Rates are significantly lower than usual; increase duration
            rationale = "Low interest rates can lead to rising bond prices; increasing duration can enhance returns."

            # Apply adjustments: increase VBLAX, reduce VBIRX
            assets_to_increase = ["VBLAX"]
            assets_to_reduce = ["VBIRX"]

            for asset in assets_to_increase:
                engine.adjust_allocation(
                    asset,
                    proposed_adjustment,
                    indicator_key,
                    rule_weight,
                    rationale,
                )

            for asset in assets_to_reduce:
                engine.adjust_allocation(
                    asset,
                    -proposed_adjustment,
                    indicator_key,
                    rule_weight,
                    rationale,
                )

            action = f"Increased allocation to VBLAX and reduced allocation to VBIRX by {proposed_adjustment:.2f}%."
            engine.logger.info(f"{rule_name}: {action} Rationale: {rationale}")

        else:
            # No significant change
            action = "No adjustment made."
            rationale = "No significant change in interest rates."

            engine.logger.info(f"{rule_name}: {action} Rationale: {rationale}")

    @staticmethod
    def apply_inflation_rule(engine: "DecisionEngine") -> None:
        """
        Adjust portfolio allocations based on changes in inflation indicators (CPI and PCE).
        """
        cpi_key = "cpi"
        pce_key = "pce"

        cpi_change = engine._validate_indicator_data(
            cpi_key, "weighted", "weighted_change"
        )
        pce_change = engine._validate_indicator_data(
            pce_key, "weighted", "weighted_change"
        )

        if cpi_change is None or pce_change is None:
            engine.logger.warning(
                "Inflation Rule: Insufficient data to calculate composite inflation change. Rule skipped."
            )
            return

        composite_change = (cpi_change * 0.5) + (pce_change * 0.5)

        upper_threshold, lower_threshold = engine.indicators[cpi_key].config.thresholds

        # Retrieve rule weight from IndicatorConfig (assuming 'cpi' is representative)
        rule_weight = engine.get_rule_weight(cpi_key)
        proposed_adjustment = min(
            engine.max_adjustment * rule_weight, abs(composite_change)
        )

        rule_name = "Inflation Adjustment"

        if composite_change > upper_threshold:
            # Allocate more to TIPS
            rationale = "High inflation reduces real returns on nominal bonds; TIPS provide inflation protection."

            # Apply adjustments: increase VTAPX, decrease VBIRX and VBLAX proportionally
            assets_to_increase = ["VTAPX"]
            assets_to_reduce = ["VBIRX", "VBLAX"]

            for asset in assets_to_increase:
                engine.adjust_allocation(
                    asset,
                    proposed_adjustment,
                    cpi_key,
                    rule_weight,
                    rationale,
                )

            reduction_per_asset = proposed_adjustment / len(assets_to_reduce)
            for asset in assets_to_reduce:
                engine.adjust_allocation(
                    asset,
                    -reduction_per_asset,
                    cpi_key,
                    rule_weight,
                    rationale,
                )

            action = f"Increased allocation to VTAPX by {proposed_adjustment:.2f}% and reduced allocations to VBIRX and VBLAX by {reduction_per_asset:.2f}% each."
            engine.logger.info(f"{rule_name}: {action} Rationale: {rationale}")

        elif composite_change < lower_threshold:
            # Allocate more to Nominal Bonds, reduce VTAPX
            rationale = "Low inflation makes nominal bonds more attractive; reducing TIPS allocation accordingly."

            # Apply adjustments: decrease VTAPX, increase VBIRX and VBLAX proportionally
            assets_to_reduce = ["VTAPX"]
            assets_to_increase = ["VBIRX", "VBLAX"]

            for asset in assets_to_reduce:
                engine.adjust_allocation(
                    asset,
                    -proposed_adjustment,
                    cpi_key,
                    rule_weight,
                    rationale,
                )

            increase_per_asset = proposed_adjustment / len(assets_to_increase)
            for asset in assets_to_increase:
                engine.adjust_allocation(
                    asset,
                    increase_per_asset,
                    cpi_key,
                    rule_weight,
                    rationale,
                )

            action = f"Decreased allocation to VTAPX by {proposed_adjustment:.2f}% and increased allocations to VBIRX and VBLAX by {increase_per_asset:.2f}% each."
            engine.logger.info(f"{rule_name}: {action} Rationale: {rationale}")

        else:
            # Stable Inflation
            action = "No adjustment made."
            rationale = "Inflation levels are stable; maintaining current allocations."

            engine.logger.info(f"{rule_name}: {action} Rationale: {rationale}")

    @staticmethod
    def apply_inflation_expectations_rule(engine: "DecisionEngine") -> None:
        """
        Adjust portfolio allocations based on changes in long-term inflation expectations.
        """
        indicator_key = "breakeven_inflation"
        change = engine._validate_indicator_data(
            indicator_key, "weighted", "weighted_change"
        )
        if change is None:
            engine.logger.warning(
                f"Inflation Expectations Rule: Unable to retrieve 'weighted_change' for indicator '{indicator_key}'. Rule skipped."
            )
            return

        upper_threshold, lower_threshold = engine.indicators[
            indicator_key
        ].config.thresholds

        # Retrieve rule weight from IndicatorConfig
        rule_weight = engine.get_rule_weight(indicator_key)
        proposed_adjustment = min(engine.max_adjustment * rule_weight, abs(change))

        rule_name = "Inflation Expectations Adjustment"

        if change > upper_threshold:
            # Inflation expectations are rising; allocate more to TIPS
            rationale = "Rising inflation expectations reduce real returns on nominal bonds; increasing TIPS hedges against inflation."

            # Apply adjustments: increase VTAPX, decrease VBIRX and VBLAX proportionally
            assets_to_increase = ["VTAPX"]
            assets_to_reduce = ["VBIRX", "VBLAX"]

            for asset in assets_to_increase:
                engine.adjust_allocation(
                    asset,
                    proposed_adjustment,
                    indicator_key,
                    rule_weight,
                    rationale,
                )

            reduction_per_asset = proposed_adjustment / len(assets_to_reduce)
            for asset in assets_to_reduce:
                engine.adjust_allocation(
                    asset,
                    -reduction_per_asset,
                    indicator_key,
                    rule_weight,
                    rationale,
                )

            action = f"Increased allocation to VTAPX by {proposed_adjustment:.2f}% and reduced allocations to VBIRX and VBLAX by {reduction_per_asset:.2f}% each."
            engine.logger.info(f"{rule_name}: {action} Rationale: {rationale}")

        elif change < lower_threshold:
            # Inflation expectations are falling; allocate more to Nominal Bonds
            rationale = "Falling inflation expectations make nominal bonds more attractive; reducing TIPS allocation accordingly."

            # Apply adjustments: decrease VTAPX, increase VBIRX and VBLAX proportionally
            assets_to_reduce = ["VTAPX"]
            assets_to_increase = ["VBIRX", "VBLAX"]

            for asset in assets_to_reduce:
                engine.adjust_allocation(
                    asset,
                    -proposed_adjustment,
                    indicator_key,
                    rule_weight,
                    rationale,
                )

            increase_per_asset = proposed_adjustment / len(assets_to_increase)
            for asset in assets_to_increase:
                engine.adjust_allocation(
                    asset,
                    increase_per_asset,
                    indicator_key,
                    rule_weight,
                    rationale,
                )

            action = f"Decreased allocation to VTAPX by {proposed_adjustment:.2f}% and increased allocations to VBIRX and VBLAX by {increase_per_asset:.2f}% each."
            engine.logger.info(f"{rule_name}: {action} Rationale: {rationale}")

        else:
            # Inflation expectations are stable
            action = "No adjustment made."
            rationale = (
                "Inflation expectations are stable; maintaining current allocations."
            )

            engine.logger.info(f"{rule_name}: {action} Rationale: {rationale}")

    @staticmethod
    def apply_yield_curve_rule(engine: "DecisionEngine") -> None:
        """
        Adjust portfolio allocations based on changes in the Yield Spread.
        """
        indicator_key = "yield_spread"
        change = engine._validate_indicator_data(
            indicator_key, "weighted", "weighted_change"
        )
        if change is None:
            engine.logger.warning(
                f"Yield Curve Rule: Unable to retrieve 'weighted_change' for indicator '{indicator_key}'. Rule skipped."
            )
            return

        upper_threshold, lower_threshold = engine.indicators[
            indicator_key
        ].config.thresholds

        # Retrieve rule weight from IndicatorConfig
        rule_weight = engine.get_rule_weight(indicator_key)
        proposed_adjustment = min(engine.max_adjustment * rule_weight, abs(change))

        rule_name = "Yield Curve Adjustment"

        if change < lower_threshold:
            # Yield spread is narrowing significantly (flattening yield curve)
            rationale = "Narrowing yield spread signals economic slowdown; long-term bonds may offer better returns."

            # Apply adjustments: increase VBLAX, decrease VBIRX
            assets_to_increase = ["VBLAX"]
            assets_to_reduce = ["VBIRX"]

            for asset in assets_to_increase:
                engine.adjust_allocation(
                    asset,
                    proposed_adjustment,
                    indicator_key,
                    rule_weight,
                    rationale,
                )

            for asset in assets_to_reduce:
                engine.adjust_allocation(
                    asset,
                    -proposed_adjustment,
                    indicator_key,
                    rule_weight,
                    rationale,
                )

            action = f"Increased allocation to VBLAX and reduced allocation to VBIRX by {proposed_adjustment:.2f}%."
            engine.logger.info(f"{rule_name}: {action} Rationale: {rationale}")

        elif change > upper_threshold:
            # Yield spread is widening significantly (steepening yield curve)
            rationale = "Widening yield spread indicates economic expansion; short-term bonds reduce interest rate risk."

            # Apply adjustments: increase VBIRX, decrease VBLAX
            assets_to_increase = ["VBIRX"]
            assets_to_reduce = ["VBLAX"]

            for asset in assets_to_increase:
                engine.adjust_allocation(
                    asset,
                    proposed_adjustment,
                    indicator_key,
                    rule_weight,
                    rationale,
                )

            for asset in assets_to_reduce:
                engine.adjust_allocation(
                    asset,
                    -proposed_adjustment,
                    indicator_key,
                    rule_weight,
                    rationale,
                )

            action = f"Increased allocation to VBIRX and reduced allocation to VBLAX by {proposed_adjustment:.2f}%."
            engine.logger.info(f"{rule_name}: {action} Rationale: {rationale}")

        else:
            # Yield spread is stable
            action = "No adjustment made."
            rationale = "Yield spread is stable, indicating no significant change in economic outlook."

            engine.logger.info(f"{rule_name}: {action} Rationale: {rationale}")

    @staticmethod
    def apply_recession_probability_rule(engine: "DecisionEngine") -> None:
        """
        Adjust portfolio allocations based on recession probabilities.
        """
        indicator_key = "recession_prob"
        current_value = engine._validate_indicator_data(
            indicator_key, "statistics", "most_recent_value"
        )
        if current_value is None:
            engine.logger.warning(
                f"Recession Probability Rule: Unable to retrieve 'most_recent_value' for indicator '{indicator_key}'. Rule skipped."
            )
            return

        upper_threshold, lower_threshold = engine.indicators[
            indicator_key
        ].config.thresholds

        # Retrieve rule weight from IndicatorConfig
        rule_weight = engine.get_rule_weight(indicator_key)
        proposed_adjustment = min(
            engine.max_adjustment * rule_weight, abs(current_value)
        )

        rule_name = "Recession Probability Adjustment"

        if current_value > upper_threshold:
            # High recession probability; increase duration
            rationale = "High recession probability may lead to lower interest rates; increasing duration can enhance returns."

            # Apply adjustments: increase VBLAX, decrease VBIRX
            assets_to_increase = ["VBLAX"]
            assets_to_reduce = ["VBIRX"]

            for asset in assets_to_increase:
                engine.adjust_allocation(
                    asset,
                    proposed_adjustment,
                    indicator_key,
                    rule_weight,
                    rationale,
                )

            for asset in assets_to_reduce:
                engine.adjust_allocation(
                    asset,
                    -proposed_adjustment,
                    indicator_key,
                    rule_weight,
                    rationale,
                )

            action = f"Increased allocation to VBLAX and reduced allocation to VBIRX by {proposed_adjustment:.2f}%."
            engine.logger.info(f"{rule_name}: {action} Rationale: {rationale}")

        elif current_value < lower_threshold:
            # Low recession probability; reduce duration
            rationale = "Low recession probability suggests rising interest rates; reducing duration mitigates interest rate risk."

            # Apply adjustments: decrease VBLAX, increase VBIRX
            assets_to_reduce = ["VBLAX"]
            assets_to_increase = ["VBIRX"]

            for asset in assets_to_reduce:
                engine.adjust_allocation(
                    asset,
                    -proposed_adjustment,
                    indicator_key,
                    rule_weight,
                    rationale,
                )

            for asset in assets_to_increase:
                engine.adjust_allocation(
                    asset,
                    proposed_adjustment,
                    indicator_key,
                    rule_weight,
                    rationale,
                )

            action = f"Decreased allocation to VBLAX and increased allocation to VBIRX by {proposed_adjustment:.2f}%."
            engine.logger.info(f"{rule_name}: {action} Rationale: {rationale}")

        else:
            # Recession probability is stable
            action = "No adjustment made."
            rationale = (
                "Recession probability is stable; maintaining current allocations."
            )

            engine.logger.info(f"{rule_name}: {action} Rationale: {rationale}")

    @staticmethod
    def apply_credit_spread_rule(engine: "DecisionEngine") -> None:
        """
        Adjust portfolio allocations based on changes in credit spreads.
        """
        indicator_key = "credit_spread"
        change = engine._validate_indicator_data(
            indicator_key, "weighted", "weighted_change"
        )
        if change is None:
            engine.logger.warning(
                f"Credit Spread Rule: Unable to retrieve 'weighted_change' for indicator '{indicator_key}'. Rule skipped."
            )
            return

        upper_threshold, lower_threshold = engine.indicators[
            indicator_key
        ].config.thresholds

        # Retrieve rule weight from IndicatorConfig
        rule_weight = engine.get_rule_weight(indicator_key)
        proposed_adjustment = min(engine.max_adjustment * rule_weight, abs(change))

        rule_name = "Credit Spread Adjustment"

        if change > upper_threshold:
            # Credit spreads have widened; reduce exposure to credit risk
            rationale = "Widening credit spreads indicate increased default risk; reducing exposure mitigates risk."

            # Apply adjustments: increase VBIRX, decrease VBLAX
            assets_to_increase = ["VBIRX"]
            assets_to_reduce = ["VBLAX"]

            for asset in assets_to_increase:
                engine.adjust_allocation(
                    asset,
                    proposed_adjustment,
                    indicator_key,
                    rule_weight,
                    rationale,
                )

            for asset in assets_to_reduce:
                engine.adjust_allocation(
                    asset,
                    -proposed_adjustment,
                    indicator_key,
                    rule_weight,
                    rationale,
                )

            action = f"Increased allocation to VBIRX and reduced allocation to VBLAX by {proposed_adjustment:.2f}%."
            engine.logger.info(f"{rule_name}: {action} Rationale: {rationale}")

        elif change < lower_threshold:
            # Credit spreads have narrowed; increase exposure to potential higher returns
            rationale = "Narrowing credit spreads indicate improved credit conditions; increasing exposure may enhance returns."

            # Apply adjustments: decrease VBIRX, increase VBLAX
            assets_to_reduce = ["VBIRX"]
            assets_to_increase = ["VBLAX"]

            for asset in assets_to_reduce:
                engine.adjust_allocation(
                    asset,
                    -proposed_adjustment,
                    indicator_key,
                    rule_weight,
                    rationale,
                )

            for asset in assets_to_increase:
                engine.adjust_allocation(
                    asset,
                    proposed_adjustment,
                    indicator_key,
                    rule_weight,
                    rationale,
                )

            action = f"Decreased allocation to VBIRX and increased allocation to VBLAX by {proposed_adjustment:.2f}%."
            engine.logger.info(f"{rule_name}: {action} Rationale: {rationale}")

        else:
            # Credit spreads are stable
            action = "No adjustment made."
            rationale = (
                "Credit spreads are stable, indicating consistent credit conditions."
            )

            engine.logger.info(f"{rule_name}: {action} Rationale: {rationale}")

    @staticmethod
    def apply_gdp_growth_rule(engine: "DecisionEngine") -> None:
        """
        Adjust portfolio allocations based on GDP growth rates.
        """
        indicator_key = "gdp"
        change = engine._validate_indicator_data(indicator_key, "1y", "change")
        if change is None:
            engine.logger.warning(
                f"GDP Growth Rule: Unable to retrieve 'change' for indicator '{indicator_key}'. Rule skipped."
            )
            return

        upper_threshold, lower_threshold = engine.indicators[
            indicator_key
        ].config.thresholds

        # Retrieve rule weight from IndicatorConfig
        rule_weight = engine.get_rule_weight(indicator_key)
        proposed_adjustment = min(engine.max_adjustment * rule_weight, abs(change))

        rule_name = "GDP Growth Adjustment"

        if change > upper_threshold:
            # GDP growth is strong; reduce duration
            rationale = "Strong GDP growth suggests economic expansion; reducing duration mitigates interest rate risk."

            # Apply adjustments: reduce VBLAX, increase VBIRX
            assets_to_reduce = ["VBLAX"]
            assets_to_increase = ["VBIRX"]

            for asset in assets_to_reduce:
                engine.adjust_allocation(
                    asset,
                    -proposed_adjustment,
                    indicator_key,
                    rule_weight,
                    rationale,
                )

            for asset in assets_to_increase:
                engine.adjust_allocation(
                    asset,
                    proposed_adjustment,
                    indicator_key,
                    rule_weight,
                    rationale,
                )

            action = f"Reduced allocation to VBLAX and increased allocation to VBIRX by {proposed_adjustment:.2f}%."
            engine.logger.info(f"{rule_name}: {action} Rationale: {rationale}")

        elif change < lower_threshold:
            # GDP growth is weak; increase duration
            rationale = "Weak GDP growth indicates economic slowdown; increasing duration favors stability."

            # Apply adjustments: increase VBLAX, decrease VBIRX
            assets_to_increase = ["VBLAX"]
            assets_to_reduce = ["VBIRX"]

            for asset in assets_to_increase:
                engine.adjust_allocation(
                    asset,
                    proposed_adjustment,
                    indicator_key,
                    rule_weight,
                    rationale,
                )

            for asset in assets_to_reduce:
                engine.adjust_allocation(
                    asset,
                    -proposed_adjustment,
                    indicator_key,
                    rule_weight,
                    rationale,
                )

            action = f"Increased allocation to VBLAX and reduced allocation to VBIRX by {proposed_adjustment:.2f}%."
            engine.logger.info(f"{rule_name}: {action} Rationale: {rationale}")

        else:
            # Stable GDP growth
            action = "No adjustment made."
            rationale = "Stable GDP growth indicates steady economic conditions."

            engine.logger.info(f"{rule_name}: {action} Rationale: {rationale}")

    @staticmethod
    def apply_employment_rule(engine: "DecisionEngine") -> None:
        """
        Adjust portfolio allocations based on changes in the unemployment rate.
        """
        indicator_key = "unrate"
        change = engine._validate_indicator_data(indicator_key, "1y", "change")
        if change is None:
            engine.logger.warning(
                f"Employment Rule: Unable to retrieve 'change' for indicator '{indicator_key}'. Rule skipped."
            )
            return

        upper_threshold, lower_threshold = engine.indicators[
            indicator_key
        ].config.thresholds

        # Retrieve rule weight from IndicatorConfig
        rule_weight = engine.get_rule_weight(indicator_key)
        proposed_adjustment = min(engine.max_adjustment * rule_weight, abs(change))

        rule_name = "Employment Rate Adjustment"

        if change > upper_threshold:
            # Unemployment rate has increased significantly; increase duration
            rationale = "Rising unemployment may indicate economic weakness; increasing duration favors safer bonds."

            # Apply adjustments: increase VBLAX, decrease VBIRX
            assets_to_increase = ["VBLAX"]
            assets_to_reduce = ["VBIRX"]

            for asset in assets_to_increase:
                engine.adjust_allocation(
                    asset,
                    proposed_adjustment,
                    indicator_key,
                    rule_weight,
                    rationale,
                )

            for asset in assets_to_reduce:
                engine.adjust_allocation(
                    asset,
                    -proposed_adjustment,
                    indicator_key,
                    rule_weight,
                    rationale,
                )

            action = f"Increased allocation to VBLAX and reduced allocation to VBIRX by {proposed_adjustment:.2f}%."
            engine.logger.info(f"{rule_name}: {action} Rationale: {rationale}")

        elif change < lower_threshold:
            # Unemployment rate has decreased significantly; reduce duration
            rationale = "Falling unemployment indicates economic strength; reducing duration mitigates interest rate risk."

            # Apply adjustments: decrease VBLAX, increase VBIRX
            assets_to_reduce = ["VBLAX"]
            assets_to_increase = ["VBIRX"]

            for asset in assets_to_reduce:
                engine.adjust_allocation(
                    asset,
                    -proposed_adjustment,
                    indicator_key,
                    rule_weight,
                    rationale,
                )

            for asset in assets_to_increase:
                engine.adjust_allocation(
                    asset,
                    proposed_adjustment,
                    indicator_key,
                    rule_weight,
                    rationale,
                )

            action = f"Decreased allocation to VBLAX and increased allocation to VBIRX by {proposed_adjustment:.2f}%."
            engine.logger.info(f"{rule_name}: {action} Rationale: {rationale}")

        else:
            # Employment levels are stable
            action = "No adjustment made."
            rationale = (
                "Employment levels are stable, suggesting steady economic conditions."
            )

            engine.logger.info(f"{rule_name}: {action} Rationale: {rationale}")
