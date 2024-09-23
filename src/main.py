# src/main.py

"""
This is the entry point for the Bondit portfolio management application. It orchestrates the workflow
of initializing loggers, loading configurations, fetching economic data, initializing the portfolio,
applying decision rules, and generating rebalancing reports.

Functions:
    main(): Executes the main workflow of the application.
"""

import sys

from .config import INDICATORS
from .decision_engine import DecisionEngine
from .utils import (
    collect_economic_indicators,
    initialize_data_storage_and_fetcher,
    initialize_logger,
    initialize_portfolio,
    load_configuration,
)


def main() -> None:
    """
    Execute the main workflow of the Bondit portfolio management application.

    The workflow includes:
        1. Initializing the main and report loggers.
        2. Loading and validating configuration settings.
        3. Initializing data storage and fetcher components.
        4. Collecting economic data and creating EconomicIndicator instances.
        5. Initializing the investment portfolio.
        6. Applying decision rules to adjust the portfolio.
        7. Generating and logging the rebalancing report.

    Raises:
        SystemExit: Exits the application if critical errors occur during initialization
                    or workflow execution.
    """
    # Initialize the main application logger
    logger = initialize_logger()
    logger.info("Logger initialized.")

    # Define the fixed configuration file path
    config_path = "config/config.yaml"
    logger.debug(f"Configuration path set to: {config_path}")

    # Define the required configuration fields
    required_config_fields = [
        "storage.fred_file_path",
        "storage.recommendation_file_path",
        "api.fred_api_key",
    ]

    # Load and validate the configuration settings
    config = load_configuration(config_path, logger, required_config_fields)

    # Initialize data storage and fetcher components
    fetcher = initialize_data_storage_and_fetcher(config, logger)

    # Step 1: Collect economic data and create EconomicIndicator instances
    indicators = collect_economic_indicators(
        fetcher, config.get("indicators", INDICATORS), logger
    )

    # Step 2: Initialize the investment portfolio
    portfolio = initialize_portfolio(logger)

    # Step 3: Apply the Decision Engine to adjust portfolio allocations
    try:
        decision_engine = DecisionEngine(
            indicators=indicators,
            portfolio=portfolio,
        )
        logger.info("Decision Engine initialized.")
        decision_engine.apply_decision_rules()
        logger.info("Decision Engine applied decision rules.")
        logger.debug(f"Portfolio after decision rules: {portfolio.get_allocations()}")
    except Exception as e:
        logger.error(f"Error applying Decision Engine rules: {e}")
        sys.exit(1)

    # Step 4: Generate and log the rebalancing report
    try:
        # Save the rebalancing report and optionally view it
        decision_engine.save_rebalancing_report(
            view=True
        )  # Set view=False if you don't want to open it automatically
        logger.info("Rebalancing report has been generated and logged.")
    except Exception as e:
        logger.error(f"Error generating or logging rebalancing report: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
