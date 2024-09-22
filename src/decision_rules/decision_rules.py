# src/decision_rules.py

"""
decision_rules.py

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
    def _create_rule_message(
        rule_name: str, analysis_details: str, action: str, rationale: str
    ) -> str:
        """
        Create a standardized message for a decision rule.

        Args:
            rule_name (str): The name of the decision rule.
            analysis_details (str): Details of the analysis that led to the decision.
            action (str): The action taken as a result of the rule.
            rationale (str): The strategic justification for the action.

        Returns:
            str: A formatted string containing the rule message.
        """
        message = (
            f"{rule_name}:\n"
            f"  - Analysis: {analysis_details}\n"
            f"  - Action: {action}\n"
            f"  - Rationale: {rationale}"
        )
        return message

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
            analysis_details = f"Federal Funds Rate z-score is {z_score:.2f}, exceeding upper threshold of {upper_threshold}."
            action = f"Reduced long-term bond allocations by {adjustment:.2f}%."
            rationale = "High interest rates can lead to falling bond prices; reducing duration mitigates risk."

            message = DecisionRules._create_rule_message(
                rule_name, analysis_details, action, rationale
            )
            engine.adjust_allocation(
                "Government Bonds - Long-Term", -adjustment, indicator_key, rule_weight
            )
            engine.adjust_allocation(
                "Corporate Bonds - Long-Term", -adjustment, indicator_key, rule_weight
            )
            engine.adjust_allocation(
                "Short-Term Bonds", adjustment * 2, indicator_key, rule_weight
            )
            engine._add_rule_message(indicator_key, message)
            engine.logger.info(message)
        elif z_score < lower_threshold:
            # Rates are significantly lower than usual; increase duration
            analysis_details = f"Federal Funds Rate z-score is {z_score:.2f}, below lower threshold of {lower_threshold}."
            action = f"Increased long-term bond allocations by {adjustment:.2f}%."
            rationale = "Low interest rates can lead to rising bond prices; increasing duration can enhance returns."

            message = DecisionRules._create_rule_message(
                rule_name, analysis_details, action, rationale
            )
            engine.adjust_allocation(
                "Government Bonds - Long-Term", adjustment, indicator_key, rule_weight
            )
            engine.adjust_allocation(
                "Corporate Bonds - Long-Term", adjustment, indicator_key, rule_weight
            )
            engine.adjust_allocation(
                "Short-Term Bonds", -adjustment * 2, indicator_key, rule_weight
            )
            engine._add_rule_message(indicator_key, message)
            engine.logger.info(message)
        else:
            # No significant change
            analysis_details = f"Federal Funds Rate z-score is {z_score:.2f}."
            action = "No adjustment made."
            rationale = "No significant change in interest rates."

            message = DecisionRules._create_rule_message(
                rule_name, analysis_details, action, rationale
            )
            engine._add_rule_message(indicator_key, message)
            engine.logger.info(message)

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
                analysis_details = f"High inflation detected with composite change {composite_change:.2f}%."
                action = f"Increased TIPS allocation by {adjustment:.2f}%."
                rationale = "High inflation reduces real returns on nominal bonds; TIPS provide inflation protection."

                message = DecisionRules._create_rule_message(
                    rule_name, analysis_details, action, rationale
                )
                engine.adjust_allocation("TIPS", adjustment, cpi_key, rule_weight)
                engine._add_rule_message(cpi_key, message)
                engine.logger.info(message)
        elif composite_change < lower_threshold:
            # Reduce TIPS, increase Nominal Bonds
            current_tips_alloc = engine.portfolio.allocations.get("TIPS", 0.0)
            adjustment = min(adjustment_amount, current_tips_alloc)

            if adjustment > 0:
                analysis_details = f"Low inflation detected with composite change {composite_change:.2f}%."
                action = f"Reduced TIPS allocation by {adjustment:.2f}% and increased Nominal Bonds by the same amount."
                rationale = "Low inflation makes nominal bonds more attractive; reducing TIPS allocation accordingly."

                message = DecisionRules._create_rule_message(
                    rule_name, analysis_details, action, rationale
                )
                engine.adjust_allocation("TIPS", -adjustment, cpi_key, rule_weight)
                engine.adjust_allocation(
                    "Nominal Bonds", adjustment, cpi_key, rule_weight
                )
                engine._add_rule_message(cpi_key, message)
                engine.logger.info(message)
        else:
            # Stable Inflation
            analysis_details = f"Stable inflation detected with composite change {composite_change:.2f}%."
            action = "No adjustment made."
            rationale = "Inflation levels are stable; maintaining current allocations."

            message = DecisionRules._create_rule_message(
                rule_name, analysis_details, action, rationale
            )
            engine._add_rule_message(cpi_key, message)
            engine.logger.info(message)

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
            analysis_details = f"Rising inflation expectations detected with a change of {change:.2f}%."
            action = f"Increased TIPS allocation by {adjustment:.2f}%."
            rationale = "Rising inflation expectations reduce real returns on nominal bonds; increasing TIPS hedges against inflation."

            message = DecisionRules._create_rule_message(
                rule_name, analysis_details, action, rationale
            )
            engine.adjust_allocation("TIPS", adjustment, indicator_key, rule_weight)
            engine._add_rule_message(indicator_key, message)
            engine.logger.info(message)
        elif change < lower_threshold:
            # Inflation expectations are falling; allocate more to Nominal Bonds
            analysis_details = f"Falling inflation expectations detected with a change of {change:.2f}%."
            action = f"Reduced TIPS allocation by {adjustment:.2f}% and increased Nominal Bonds by the same amount."
            rationale = "Falling inflation expectations make nominal bonds more attractive; reducing TIPS allocation accordingly."

            message = DecisionRules._create_rule_message(
                rule_name, analysis_details, action, rationale
            )
            engine.adjust_allocation("TIPS", -adjustment, indicator_key, rule_weight)
            engine.adjust_allocation(
                "Nominal Bonds", adjustment, indicator_key, rule_weight
            )
            engine._add_rule_message(indicator_key, message)
            engine.logger.info(message)
        else:
            # Inflation expectations are stable
            analysis_details = f"Inflation expectations change is {change:.2f}%."
            action = "No adjustment made."
            rationale = (
                "Inflation expectations are stable; maintaining current allocations."
            )

            message = DecisionRules._create_rule_message(
                rule_name, analysis_details, action, rationale
            )
            engine._add_rule_message(indicator_key, message)
            engine.logger.info(message)

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
            analysis_details = f"Yield spread narrowing by {change:.2f}%, below lower threshold of {lower_threshold}%."
            action = f"Increased allocation to long-term government bonds by {adjustment:.2f}%."
            rationale = "Narrowing yield spread signals economic slowdown; long-term bonds offer safety."

            message = DecisionRules._create_rule_message(
                rule_name, analysis_details, action, rationale
            )
            engine.adjust_allocation(
                "Government Bonds - Long-Term", adjustment, indicator_key, rule_weight
            )
            engine._add_rule_message(indicator_key, message)
            engine.logger.info(message)
        elif change > upper_threshold:
            # Yield spread is widening significantly
            analysis_details = f"Yield spread widening by {change:.2f}%, above upper threshold of {upper_threshold}%."
            action = (
                f"Reduced long-term government bond allocation by {adjustment:.2f}%."
            )
            rationale = "Widening yield spread indicates economic expansion; reducing duration mitigates risk."

            message = DecisionRules._create_rule_message(
                rule_name, analysis_details, action, rationale
            )
            engine.adjust_allocation(
                "Government Bonds - Long-Term", -adjustment, indicator_key, rule_weight
            )
            engine.adjust_allocation(
                "Short-Term Bonds", adjustment, indicator_key, rule_weight
            )
            engine._add_rule_message(indicator_key, message)
            engine.logger.info(message)
        else:
            # Yield spread is stable
            analysis_details = f"Yield spread change is {change:.2f}%."
            action = "No adjustment made."
            rationale = "Yield spread is stable, indicating no significant change in economic outlook."

            message = DecisionRules._create_rule_message(
                rule_name, analysis_details, action, rationale
            )
            engine._add_rule_message(indicator_key, message)
            engine.logger.info(message)

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
            analysis_details = f"High recession probability detected at {current_value:.2f}%, exceeding upper threshold of {upper_threshold}%."
            action = (
                f"Increased allocation to government bonds by {adjustment / 2:.2f}%."
            )
            rationale = "High recession probability increases default risk on corporate bonds; shifting to safer government bonds reduces credit risk."

            message = DecisionRules._create_rule_message(
                rule_name, analysis_details, action, rationale
            )
            engine.adjust_allocation(
                "Corporate Bonds - Long-Term",
                -adjustment / 2,
                indicator_key,
                rule_weight,
            )
            engine.adjust_allocation(
                "Corporate Bonds - Intermediate-Term",
                -adjustment / 2,
                indicator_key,
                rule_weight,
            )
            engine.adjust_allocation(
                "Government Bonds - Long-Term",
                adjustment / 2,
                indicator_key,
                rule_weight,
            )
            engine.adjust_allocation(
                "Government Bonds - Intermediate-Term",
                adjustment / 2,
                indicator_key,
                rule_weight,
            )
            engine._add_rule_message(indicator_key, message)
            engine.logger.info(message)
        elif current_value < lower_threshold:
            # Low recession probability; increase allocation to corporate bonds
            analysis_details = f"Low recession probability detected at {current_value:.2f}%, below lower threshold of {lower_threshold}%."
            action = (
                f"Increased allocation to corporate bonds by {adjustment / 2:.2f}%."
            )
            rationale = "Low recession probability reduces default risk on corporate bonds; increasing exposure can enhance returns."

            message = DecisionRules._create_rule_message(
                rule_name, analysis_details, action, rationale
            )
            engine.adjust_allocation(
                "Government Bonds - Long-Term",
                -adjustment / 2,
                indicator_key,
                rule_weight,
            )
            engine.adjust_allocation(
                "Government Bonds - Intermediate-Term",
                -adjustment / 2,
                indicator_key,
                rule_weight,
            )
            engine.adjust_allocation(
                "Corporate Bonds - Long-Term",
                adjustment / 2,
                indicator_key,
                rule_weight,
            )
            engine.adjust_allocation(
                "Corporate Bonds - Intermediate-Term",
                adjustment / 2,
                indicator_key,
                rule_weight,
            )
            engine._add_rule_message(indicator_key, message)
            engine.logger.info(message)
        else:
            # Recession probability is stable
            analysis_details = f"Recession probability is {current_value:.2f}%."
            action = "No adjustment made."
            rationale = (
                "Recession probability is stable; maintaining current allocations."
            )

            message = DecisionRules._create_rule_message(
                rule_name, analysis_details, action, rationale
            )
            engine._add_rule_message(indicator_key, message)
            engine.logger.info(message)

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
            analysis_details = f"Credit spreads have widened by {change:.2f}%, exceeding upper threshold of {upper_threshold}%."
            action = f"Reduced corporate bond exposure by {adjustment / 2:.2f}%."
            rationale = "Widening credit spreads indicate increased default risk; reducing exposure to corporate bonds lowers credit risk."

            message = DecisionRules._create_rule_message(
                rule_name, analysis_details, action, rationale
            )
            engine.adjust_allocation(
                "Corporate Bonds - Long-Term",
                -adjustment / 2,
                indicator_key,
                rule_weight,
            )
            engine.adjust_allocation(
                "Corporate Bonds - Intermediate-Term",
                -adjustment / 2,
                indicator_key,
                rule_weight,
            )
            engine.adjust_allocation(
                "Government Bonds - Long-Term",
                adjustment / 2,
                indicator_key,
                rule_weight,
            )
            engine.adjust_allocation(
                "Government Bonds - Intermediate-Term",
                adjustment / 2,
                indicator_key,
                rule_weight,
            )
            engine._add_rule_message(indicator_key, message)
            engine.logger.info(message)
        elif change < lower_threshold:
            # Credit spreads have narrowed; increase exposure to corporate bonds
            analysis_details = f"Credit spreads have narrowed by {change:.2f}%, below lower threshold of {lower_threshold}%."
            action = f"Increased corporate bond exposure by {adjustment / 2:.2f}%."
            rationale = "Narrowing credit spreads indicate improved corporate credit conditions; increasing exposure can enhance returns."

            message = DecisionRules._create_rule_message(
                rule_name, analysis_details, action, rationale
            )
            engine.adjust_allocation(
                "Government Bonds - Long-Term",
                -adjustment / 2,
                indicator_key,
                rule_weight,
            )
            engine.adjust_allocation(
                "Government Bonds - Intermediate-Term",
                -adjustment / 2,
                indicator_key,
                rule_weight,
            )
            engine.adjust_allocation(
                "Corporate Bonds - Long-Term",
                adjustment / 2,
                indicator_key,
                rule_weight,
            )
            engine.adjust_allocation(
                "Corporate Bonds - Intermediate-Term",
                adjustment / 2,
                indicator_key,
                rule_weight,
            )
            engine._add_rule_message(indicator_key, message)
            engine.logger.info(message)
        else:
            # Credit spread is stable
            analysis_details = f"Credit spread change is {change:.2f}%."
            action = "No adjustment made."
            rationale = "Credit spreads are stable, indicating consistent corporate credit conditions."

            message = DecisionRules._create_rule_message(
                rule_name, analysis_details, action, rationale
            )
            engine._add_rule_message(indicator_key, message)
            engine.logger.info(message)

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
            analysis_details = f"GDP growth increased by {change:.2f}%, exceeding the upper threshold of {upper_threshold}."
            action = f"Reduced long-term bond allocations by {adjustment:.2f}%."
            rationale = "Strong GDP growth suggests economic expansion; reducing duration mitigates interest rate risk."

            message = DecisionRules._create_rule_message(
                rule_name, analysis_details, action, rationale
            )
            engine.adjust_allocation(
                "Government Bonds - Long-Term", -adjustment, indicator_key, rule_weight
            )
            engine.adjust_allocation(
                "Corporate Bonds - Long-Term", -adjustment, indicator_key, rule_weight
            )
            engine.adjust_allocation(
                "Short-Term Bonds", adjustment * 2, indicator_key, rule_weight
            )
            engine._add_rule_message(indicator_key, message)
            engine.logger.info(message)
        elif change < lower_threshold:
            # GDP growth is weak; increase duration
            analysis_details = f"GDP growth decreased by {change:.2f}%, below the lower threshold of {lower_threshold}."
            action = f"Increased long-term bond allocations by {adjustment:.2f}%."
            rationale = "Weak GDP growth indicates economic slowdown; increasing duration favors stability."

            message = DecisionRules._create_rule_message(
                rule_name, analysis_details, action, rationale
            )
            engine.adjust_allocation(
                "Government Bonds - Long-Term", adjustment, indicator_key, rule_weight
            )
            engine.adjust_allocation(
                "Corporate Bonds - Long-Term", adjustment, indicator_key, rule_weight
            )
            engine.adjust_allocation(
                "Short-Term Bonds", -adjustment * 2, indicator_key, rule_weight
            )
            engine._add_rule_message(indicator_key, message)
            engine.logger.info(message)
        else:
            # GDP growth is stable
            analysis_details = f"GDP growth change is {change:.2f}%."
            action = "No adjustment made."
            rationale = "Stable GDP growth indicates steady economic conditions."

            message = DecisionRules._create_rule_message(
                rule_name, analysis_details, action, rationale
            )
            engine._add_rule_message(indicator_key, message)
            engine.logger.info(message)

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
            analysis_details = f"Unemployment rate increased by {change:.2f}%, exceeding upper threshold of {upper_threshold}%."
            action = f"Increased long-term bond allocations by {adjustment:.2f}%."
            rationale = "Rising unemployment may indicate economic weakness; increasing duration favors safer bonds."

            message = DecisionRules._create_rule_message(
                rule_name, analysis_details, action, rationale
            )
            engine.adjust_allocation(
                "Government Bonds - Long-Term", adjustment, indicator_key, rule_weight
            )
            engine.adjust_allocation(
                "Corporate Bonds - Long-Term", adjustment, indicator_key, rule_weight
            )
            engine.adjust_allocation(
                "Short-Term Bonds", -adjustment * 2, indicator_key, rule_weight
            )
            engine._add_rule_message(indicator_key, message)
            engine.logger.info(message)
        elif change < lower_threshold:
            # Unemployment rate has decreased significantly; reduce duration
            analysis_details = f"Unemployment rate decreased by {change:.2f}%, below lower threshold of {lower_threshold}%."
            action = f"Reduced long-term bond allocations by {adjustment:.2f}%."
            rationale = "Falling unemployment indicates economic strength; reducing duration mitigates interest rate risk."

            message = DecisionRules._create_rule_message(
                rule_name, analysis_details, action, rationale
            )
            engine.adjust_allocation(
                "Government Bonds - Long-Term", -adjustment, indicator_key, rule_weight
            )
            engine.adjust_allocation(
                "Corporate Bonds - Long-Term", -adjustment, indicator_key, rule_weight
            )
            engine.adjust_allocation(
                "Short-Term Bonds", adjustment * 2, indicator_key, rule_weight
            )
            engine._add_rule_message(indicator_key, message)
            engine.logger.info(message)
        else:
            # Unemployment rate is stable
            analysis_details = f"Unemployment rate change is {change:.2f}%."
            action = "No adjustment made."
            rationale = (
                "Employment levels are stable, suggesting steady economic conditions."
            )

            message = DecisionRules._create_rule_message(
                rule_name, analysis_details, action, rationale
            )
            engine._add_rule_message(indicator_key, message)
            engine.logger.info(message)
