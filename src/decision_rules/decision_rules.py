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
        adjustment = min(engine.max_adjustment * rule_weight, abs(z_score))

        rule_name = "Interest Rate Adjustment"

        if z_score > upper_threshold:
            # Rates are significantly higher than usual; reduce duration
            action = f"Reduced long-term bond allocations by {adjustment:.2f}%."
            rationale = "High interest rates can lead to falling bond prices; reducing duration mitigates risk."

            # Adjust allocations
            engine.adjust_allocation(
                "Government Bonds - Long-Term",
                -adjustment,
                indicator_key,
                rule_weight,
                rationale,
            )
            engine.adjust_allocation(
                "Corporate Bonds - Long-Term",
                -adjustment,
                indicator_key,
                rule_weight,
                rationale,
            )
            engine.adjust_allocation(
                "Short-Term Bonds",
                adjustment * 2,
                indicator_key,
                rule_weight,
                rationale,
            )

            engine.logger.info(f"{rule_name}: {action} Rationale: {rationale}")
        elif z_score < lower_threshold:
            # Rates are significantly lower than usual; increase duration
            action = f"Increased long-term bond allocations by {adjustment:.2f}%."
            rationale = "Low interest rates can lead to rising bond prices; increasing duration can enhance returns."

            # Adjust allocations
            engine.adjust_allocation(
                "Government Bonds - Long-Term",
                adjustment,
                indicator_key,
                rule_weight,
                rationale,
            )
            engine.adjust_allocation(
                "Corporate Bonds - Long-Term",
                adjustment,
                indicator_key,
                rule_weight,
                rationale,
            )
            engine.adjust_allocation(
                "Short-Term Bonds",
                -adjustment * 2,
                indicator_key,
                rule_weight,
                rationale,
            )

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
        adjustment_amount = engine.max_adjustment * rule_weight

        rule_name = "Inflation Adjustment"

        if composite_change > upper_threshold:
            # Allocate more to TIPS
            desired_tips_alloc = 20.0
            current_tips_alloc = engine.portfolio.allocations.get("TIPS", 0.0)
            adjustment = min(adjustment_amount, desired_tips_alloc - current_tips_alloc)

            if adjustment > 0:
                action = f"Increased TIPS allocation by {adjustment:.2f}%."
                rationale = "High inflation reduces real returns on nominal bonds; TIPS provide inflation protection."

                engine.adjust_allocation(
                    "TIPS", adjustment, cpi_key, rule_weight, rationale
                )

                engine.logger.info(f"{rule_name}: {action} Rationale: {rationale}")
        elif composite_change < lower_threshold:
            # Reduce TIPS, increase Nominal Bonds
            current_tips_alloc = engine.portfolio.allocations.get("TIPS", 0.0)
            adjustment = min(adjustment_amount, current_tips_alloc)

            if adjustment > 0:
                action = f"Reduced TIPS allocation by {adjustment:.2f}% and increased Nominal Bonds by the same amount."
                rationale = "Low inflation makes nominal bonds more attractive; reducing TIPS allocation accordingly."

                engine.adjust_allocation(
                    "TIPS", -adjustment, cpi_key, rule_weight, rationale
                )
                engine.adjust_allocation(
                    "Nominal Bonds", adjustment, cpi_key, rule_weight, rationale
                )

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
        adjustment = min(engine.max_adjustment * rule_weight, abs(change))

        rule_name = "Inflation Expectations Adjustment"

        if change > upper_threshold:
            # Inflation expectations are rising; allocate more to TIPS
            action = f"Increased TIPS allocation by {adjustment:.2f}%."
            rationale = "Rising inflation expectations reduce real returns on nominal bonds; increasing TIPS hedges against inflation."

            engine.adjust_allocation(
                "TIPS", adjustment, indicator_key, rule_weight, rationale
            )

            engine.logger.info(f"{rule_name}: {action} Rationale: {rationale}")
        elif change < lower_threshold:
            # Inflation expectations are falling; allocate more to Nominal Bonds
            action = f"Reduced TIPS allocation by {adjustment:.2f}% and increased Nominal Bonds by the same amount."
            rationale = "Falling inflation expectations make nominal bonds more attractive; reducing TIPS allocation accordingly."

            engine.adjust_allocation(
                "TIPS", -adjustment, indicator_key, rule_weight, rationale
            )
            engine.adjust_allocation(
                "Nominal Bonds", adjustment, indicator_key, rule_weight, rationale
            )

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
        adjustment = min(engine.max_adjustment * rule_weight, abs(change))

        rule_name = "Yield Curve Adjustment"

        if change < lower_threshold:
            # Yield spread is narrowing significantly
            action = f"Increased allocation to long-term government bonds by {adjustment:.2f}%."
            rationale = "Narrowing yield spread signals economic slowdown; long-term bonds offer safety."

            engine.adjust_allocation(
                "Government Bonds - Long-Term",
                adjustment,
                indicator_key,
                rule_weight,
                rationale,
            )

            engine.logger.info(f"{rule_name}: {action} Rationale: {rationale}")
        elif change > upper_threshold:
            # Yield spread is widening significantly
            action = (
                f"Reduced long-term government bond allocation by {adjustment:.2f}%."
            )
            rationale = "Widening yield spread indicates economic expansion; reducing duration mitigates risk."

            engine.adjust_allocation(
                "Government Bonds - Long-Term",
                -adjustment,
                indicator_key,
                rule_weight,
                rationale,
            )
            engine.adjust_allocation(
                "Short-Term Bonds", adjustment, indicator_key, rule_weight, rationale
            )

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
        adjustment = engine.max_adjustment * rule_weight

        rule_name = "Recession Probability Adjustment"

        if current_value > upper_threshold:
            # High recession probability; increase allocation to government bonds
            action = (
                f"Increased allocation to government bonds by {adjustment / 2:.2f}%."
            )
            rationale = "High recession probability increases default risk on corporate bonds; shifting to safer government bonds reduces credit risk."

            engine.adjust_allocation(
                "Corporate Bonds - Long-Term",
                -adjustment / 2,
                indicator_key,
                rule_weight,
                rationale,
            )
            engine.adjust_allocation(
                "Corporate Bonds - Intermediate-Term",
                -adjustment / 2,
                indicator_key,
                rule_weight,
                rationale,
            )
            engine.adjust_allocation(
                "Government Bonds - Long-Term",
                adjustment / 2,
                indicator_key,
                rule_weight,
                rationale,
            )
            engine.adjust_allocation(
                "Government Bonds - Intermediate-Term",
                adjustment / 2,
                indicator_key,
                rule_weight,
                rationale,
            )

            engine.logger.info(f"{rule_name}: {action} Rationale: {rationale}")
        elif current_value < lower_threshold:
            # Low recession probability; increase allocation to corporate bonds
            action = (
                f"Increased allocation to corporate bonds by {adjustment / 2:.2f}%."
            )
            rationale = "Low recession probability reduces default risk on corporate bonds; increasing exposure can enhance returns."

            engine.adjust_allocation(
                "Government Bonds - Long-Term",
                -adjustment / 2,
                indicator_key,
                rule_weight,
                rationale,
            )
            engine.adjust_allocation(
                "Government Bonds - Intermediate-Term",
                -adjustment / 2,
                indicator_key,
                rule_weight,
                rationale,
            )
            engine.adjust_allocation(
                "Corporate Bonds - Long-Term",
                adjustment / 2,
                indicator_key,
                rule_weight,
                rationale,
            )
            engine.adjust_allocation(
                "Corporate Bonds - Intermediate-Term",
                adjustment / 2,
                indicator_key,
                rule_weight,
                rationale,
            )

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
        adjustment = min(engine.max_adjustment * rule_weight, abs(change))

        rule_name = "Credit Spread Adjustment"

        if change > upper_threshold:
            # Credit spreads have widened; reduce exposure to corporate bonds
            action = f"Reduced corporate bond exposure by {adjustment / 2:.2f}%."
            rationale = "Widening credit spreads indicate increased default risk; reducing exposure to corporate bonds lowers credit risk."

            engine.adjust_allocation(
                "Corporate Bonds - Long-Term",
                -adjustment / 2,
                indicator_key,
                rule_weight,
                rationale,
            )
            engine.adjust_allocation(
                "Corporate Bonds - Intermediate-Term",
                -adjustment / 2,
                indicator_key,
                rule_weight,
                rationale,
            )
            engine.adjust_allocation(
                "Government Bonds - Long-Term",
                adjustment / 2,
                indicator_key,
                rule_weight,
                rationale,
            )
            engine.adjust_allocation(
                "Government Bonds - Intermediate-Term",
                adjustment / 2,
                indicator_key,
                rule_weight,
                rationale,
            )

            engine.logger.info(f"{rule_name}: {action} Rationale: {rationale}")
        elif change < lower_threshold:
            # Credit spreads have narrowed; increase exposure to corporate bonds
            action = f"Increased corporate bond exposure by {adjustment / 2:.2f}%."
            rationale = "Narrowing credit spreads indicate improved corporate credit conditions; increasing exposure can enhance returns."

            engine.adjust_allocation(
                "Government Bonds - Long-Term",
                -adjustment / 2,
                indicator_key,
                rule_weight,
                rationale,
            )
            engine.adjust_allocation(
                "Government Bonds - Intermediate-Term",
                -adjustment / 2,
                indicator_key,
                rule_weight,
                rationale,
            )
            engine.adjust_allocation(
                "Corporate Bonds - Long-Term",
                adjustment / 2,
                indicator_key,
                rule_weight,
                rationale,
            )
            engine.adjust_allocation(
                "Corporate Bonds - Intermediate-Term",
                adjustment / 2,
                indicator_key,
                rule_weight,
                rationale,
            )

            engine.logger.info(f"{rule_name}: {action} Rationale: {rationale}")
        else:
            # Credit spread is stable
            action = "No adjustment made."
            rationale = "Credit spreads are stable, indicating consistent corporate credit conditions."

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
        adjustment = min(engine.max_adjustment * rule_weight, abs(change))

        rule_name = "GDP Growth Adjustment"

        if change > upper_threshold:
            # GDP growth is strong; reduce duration
            action = f"Reduced long-term bond allocations by {adjustment:.2f}%."
            rationale = "Strong GDP growth suggests economic expansion; reducing duration mitigates interest rate risk."

            engine.adjust_allocation(
                "Government Bonds - Long-Term",
                -adjustment,
                indicator_key,
                rule_weight,
                rationale,
            )
            engine.adjust_allocation(
                "Corporate Bonds - Long-Term",
                -adjustment,
                indicator_key,
                rule_weight,
                rationale,
            )
            engine.adjust_allocation(
                "Short-Term Bonds",
                adjustment * 2,
                indicator_key,
                rule_weight,
                rationale,
            )

            engine.logger.info(f"{rule_name}: {action} Rationale: {rationale}")
        elif change < lower_threshold:
            # GDP growth is weak; increase duration
            action = f"Increased long-term bond allocations by {adjustment:.2f}%."
            rationale = "Weak GDP growth indicates economic slowdown; increasing duration favors stability."

            engine.adjust_allocation(
                "Government Bonds - Long-Term",
                adjustment,
                indicator_key,
                rule_weight,
                rationale,
            )
            engine.adjust_allocation(
                "Corporate Bonds - Long-Term",
                adjustment,
                indicator_key,
                rule_weight,
                rationale,
            )
            engine.adjust_allocation(
                "Short-Term Bonds",
                -adjustment * 2,
                indicator_key,
                rule_weight,
                rationale,
            )

            engine.logger.info(f"{rule_name}: {action} Rationale: {rationale}")
        else:
            # GDP growth is stable
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
        adjustment = min(engine.max_adjustment * rule_weight, abs(change))

        rule_name = "Employment Rate Adjustment"

        if change > upper_threshold:
            # Unemployment rate has increased significantly; increase duration
            action = f"Increased long-term bond allocations by {adjustment:.2f}%."
            rationale = "Rising unemployment may indicate economic weakness; increasing duration favors safer bonds."

            engine.adjust_allocation(
                "Government Bonds - Long-Term",
                adjustment,
                indicator_key,
                rule_weight,
                rationale,
            )
            engine.adjust_allocation(
                "Corporate Bonds - Long-Term",
                adjustment,
                indicator_key,
                rule_weight,
                rationale,
            )
            engine.adjust_allocation(
                "Short-Term Bonds",
                -adjustment * 2,
                indicator_key,
                rule_weight,
                rationale,
            )

            engine.logger.info(f"{rule_name}: {action} Rationale: {rationale}")
        elif change < lower_threshold:
            # Unemployment rate has decreased significantly; reduce duration
            action = f"Reduced long-term bond allocations by {adjustment:.2f}%."
            rationale = "Falling unemployment indicates economic strength; reducing duration mitigates interest rate risk."

            engine.adjust_allocation(
                "Government Bonds - Long-Term",
                -adjustment,
                indicator_key,
                rule_weight,
                rationale,
            )
            engine.adjust_allocation(
                "Corporate Bonds - Long-Term",
                -adjustment,
                indicator_key,
                rule_weight,
                rationale,
            )
            engine.adjust_allocation(
                "Short-Term Bonds",
                adjustment * 2,
                indicator_key,
                rule_weight,
                rationale,
            )

            engine.logger.info(f"{rule_name}: {action} Rationale: {rationale}")
        else:
            # Unemployment rate is stable
            action = "No adjustment made."
            rationale = (
                "Employment levels are stable, suggesting steady economic conditions."
            )

            engine.logger.info(f"{rule_name}: {action} Rationale: {rationale}")
