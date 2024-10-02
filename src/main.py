# src/main.py

"""
This is the entry point for the Bondit portfolio management application. It orchestrates the workflow
of initializing loggers, loading configurations, fetching economic data, initializing the portfolio,
applying decision rules, and generating rebalancing reports.

Functions:
    main(): Executes the main workflow of the application.
"""

import sys
from datetime import datetime
from typing import Optional

# Import INDICATORS from config.py
from .config import INDICATORS
from .decision_engine import DecisionEngine
from .portfolio import Portfolio
from .utils import (
    collect_economic_indicators,
    initialize_data_storage_and_fetcher,
    initialize_logger,
    load_configuration,
    parse_allocations,
)


def main(rebalancing_date: Optional[str] = None) -> None:
    """
    Execute the main workflow of the Bondit portfolio management application.

    The workflow includes:
        1. Initializing the main logger.
        2. Loading and validating configuration settings.
        3. Initializing data storage and fetcher components.
        4. Collecting economic data and creating EconomicIndicator instances.
        5. Initializing the investment portfolio.
        6. Applying decision rules to adjust the portfolio.
        7. Generating and logging the rebalancing report.

    Args:
        rebalancing_date (Optional[str], optional): The date up to which data should be analyzed in "YYYY-MM-DD" format.
                                                    Defaults to today's date if not provided.

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
        "allocations",  # Added allocations as a required field
    ]

    # Load and validate the configuration settings
    try:
        config = load_configuration(config_path, logger, required_config_fields)
        logger.info("Configuration loaded successfully.")
    except Exception as e:
        logger.critical(f"Failed to load configuration: {e}")
        sys.exit(1)

    # **Specify Rebalancing Date in main.py**
    if rebalancing_date:
        try:
            # Validate the format of the rebalancing date
            datetime.strptime(rebalancing_date, "%Y-%m-%d")
            logger.info(f"Using specified rebalancing date: {rebalancing_date}.")
        except ValueError:
            logger.error(
                f"Invalid rebalancing date format: {rebalancing_date}. Expected 'YYYY-MM-DD'."
            )
            sys.exit(1)
    else:
        # Default to today's date if not specified
        rebalancing_date = datetime.now().strftime("%Y-%m-%d")
        logger.info(
            f"No rebalancing date specified. Defaulting to today: {rebalancing_date}."
        )

    # Initialize data storage and fetcher components
    try:
        fetcher = initialize_data_storage_and_fetcher(config, logger)
        logger.info("Data storage and fetcher initialized.")
    except Exception as e:
        logger.critical(f"Failed to initialize data storage and fetcher: {e}")
        sys.exit(1)

    # Step 1: Collect economic data and create EconomicIndicator instances
    try:
        indicators = collect_economic_indicators(
            fetcher,
            INDICATORS,  # Pass INDICATORS directly
            logger,
            rebalancing_date=rebalancing_date,  # Pass the rebalancing date
        )
        logger.info("Economic indicators collected successfully.")
    except Exception as e:
        logger.critical(f"Failed to collect economic indicators: {e}")
        sys.exit(1)

    # Step 2: Initialize the investment portfolio with allocation constraints
    try:
        (
            initial_allocations,
            min_allocations,
            max_allocations,
            allocation_constraints,
        ) = parse_allocations(config, logger)

        # Initialize the Portfolio with the parsed allocations
        portfolio = Portfolio(
            allocations=initial_allocations,
            min_allocations=min_allocations,
            max_allocations=max_allocations,
            logger=logger,
        )
        logger.info("Portfolio initialized successfully.")
    except ValueError as e:
        logger.critical(f"Failed to initialize Portfolio: {e}")
        sys.exit(1)
    except KeyError as e:
        logger.critical(f"Missing allocation key in configuration: {e}")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"Unexpected error during Portfolio initialization: {e}")
        sys.exit(1)

    # Step 3: Apply the Decision Engine to adjust portfolio allocations
    try:
        decision_engine = DecisionEngine(
            indicators=indicators,
            portfolio=portfolio,
            allocation_constraints=allocation_constraints,  # Now correctly defined
            logger=logger,
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
    # **Optional: Accept Rebalancing Date as a Command-Line Argument**
    # This allows for easier backtesting by passing different dates when running the script.
    import argparse

    parser = argparse.ArgumentParser(
        description="Bondit Portfolio Management Application"
    )
    parser.add_argument(
        "--rebalancing_date",
        type=str,
        help="Rebalancing date in 'YYYY-MM-DD' format. Defaults to today if not provided.",
    )
    args = parser.parse_args()

    main(rebalancing_date=args.rebalancing_date)
