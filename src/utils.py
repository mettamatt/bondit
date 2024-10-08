# src/utils.py

"""
This module provides utility functions to support the Bondit portfolio management application.
It includes functions for initializing loggers, loading configurations, initializing data storage
and fetchers, collecting economic indicators, and initializing the investment portfolio.

Functions:
    initialize_logger(logger_name: str = "Bondit") -> logging.Logger:
        Configures and returns the main application logger.

    initialize_report_logger(report_logger_name: str = "Bondit.RebalancingReport",
                            log_file: str = "rebalancing_report.log") -> logging.Logger:
        Configures and returns a dedicated logger for rebalancing reports.

    load_configuration(config_path: str, logger: logging.Logger,
                      required_fields: List[str]) -> Dict[str, Any]:
        Loads and validates the YAML configuration file.

    initialize_data_storage_and_fetcher(config: Dict[str, Any],
                                       logger: logging.Logger) -> DataFetcher:
        Initializes data storage and fetcher components.

    collect_economic_indicators(fetcher: DataFetcher,
                                indicators_config: List[IndicatorConfig],
                                logger: logging.Logger,
                                rebalancing_date: Optional[str] = None) -> Dict[str, EconomicIndicator]:
        Collects economic data and creates EconomicIndicator instances.

    initialize_portfolio(logger: logging.Logger) -> Portfolio:
        Initializes and returns the investment portfolio.
"""

import logging
import math
import os
import sys
from typing import Any, Dict, List, Optional, Tuple, cast

import yaml

from .config import IndicatorConfig
from .data_fetcher import DataFetcher
from .data_storage import FredDataStorage
from .indicators import EconomicIndicator
from .portfolio import Portfolio


def initialize_logger(logger_name: str = "Bondit") -> logging.Logger:
    """
    Initialize and configure the main application logger.

    This function sets up the primary logger for the application, configuring it to output
    debug and higher-level messages to the console. It also adjusts the log levels of
    specific child loggers to reduce verbosity where necessary.

    Args:
        logger_name (str, optional): The name of the logger. Defaults to "Bondit".

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)  # Set the logger to capture all levels

    if not logger.handlers:
        # Console handler for real-time output
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)  # Handler captures INFO and higher levels

        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        ch.setFormatter(formatter)

        logger.addHandler(ch)

    # Adjust specific child loggers to reduce verbosity
    child_loggers = [
        "Bondit.DataStorage",
        "Bondit.DataFetcher",
        "Bondit.EconomicIndicator",
        "Bondit.DecisionEngine",
        "Bondit.Portfolio",
    ]
    for child in child_loggers:
        logging.getLogger(child).setLevel(logging.WARNING)

    return logger


def load_configuration(
    config_path: str, logger: logging.Logger, required_fields: List[str]
) -> Dict[str, Any]:
    """
    Load and validate the YAML configuration file.

    This function reads the configuration from a YAML file, ensures that all required fields
    are present, and logs relevant information. If critical issues are encountered, the
    application exits gracefully.

    Args:
        config_path (str): The path to the YAML configuration file.
        logger (logging.Logger): Logger instance for logging.
        required_fields (List[str]): List of required configuration fields in dot notation.

    Returns:
        Dict[str, Any]: Loaded configuration dictionary.

    Raises:
        SystemExit: If the configuration file is missing, malformed, or lacks required fields.
    """
    logger.debug(f"Loading configuration from: {config_path}")

    if not os.path.exists(config_path):
        logger.critical(f"Configuration file not found at: {config_path}")
        sys.exit(1)

    try:
        with open(config_path, "r") as file:
            raw_config = yaml.safe_load(file)
            if not isinstance(raw_config, dict):
                logger.critical(
                    "Configuration file does not contain a valid dictionary."
                )
                sys.exit(1)
            config: Dict[str, Any] = cast(Dict[str, Any], raw_config)
            logger.info("Configuration loaded successfully.")
    except yaml.YAMLError as e:
        logger.critical(f"Error parsing YAML configuration: {e}")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"Unexpected error loading configuration: {e}")
        sys.exit(1)

    # Validate the presence of required configuration fields
    missing_fields = []
    for field in required_fields:
        keys = field.split(".")
        temp = config
        for key in keys:
            if key not in temp:
                missing_fields.append(field)
                break
            temp = temp[key]

    if missing_fields:
        for field in missing_fields:
            logger.error(f"Missing configuration field: {field}")
        sys.exit(1)
    else:
        logger.info("All required configuration fields are present.")

    return config


def initialize_data_storage_and_fetcher(
    config: Dict[str, Any], logger: logging.Logger
) -> DataFetcher:
    """
    Initialize data storage and fetcher components.

    This function sets up the data storage using FredDataStorage and initializes the
    DataFetcher with the appropriate API key. It logs relevant information and exits
    the application if initialization fails.

    Args:
        config (Dict[str, Any]): Loaded configuration dictionary.
        logger (logging.Logger): Logger instance for logging.

    Returns:
        DataFetcher: Initialized DataFetcher instance.

    Raises:
        SystemExit: If initialization of data storage or fetcher fails.
    """
    try:
        fred_file_path = config["storage"]["fred_file_path"]
        fred_api_key = config["api"]["fred_api_key"]

        if not fred_api_key:
            logger.critical("FRED API key is missing in the configuration.")
            sys.exit(1)

        storage = FredDataStorage(fred_file_path, logger=logger)
        fetcher = DataFetcher(storage, fred_api_key, logger_instance=logger)
        logger.info("Initialized Data Storage and Data Fetcher.")
        logger.debug(f"Storage path: {fred_file_path}")
        logger.debug(f"FRED API Key: {'***' if fred_api_key else 'None'}")
        return fetcher
    except KeyError as e:
        logger.critical(f"Missing configuration key during initialization: {e}")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"Initialization error: {e}", exc_info=True)
        sys.exit(1)


def collect_economic_indicators(
    fetcher: DataFetcher,
    indicators_config: List[IndicatorConfig],
    logger: logging.Logger,
    rebalancing_date: Optional[str] = None,
) -> Dict[str, EconomicIndicator]:
    """
    Collect economic data and create EconomicIndicator instances.

    This function iterates through the list of economic indicators defined in the configuration,
    fetches their data up to the specified rebalancing date, validates the fetched data,
    and creates corresponding EconomicIndicator instances. It logs successes and
    handles any data validation issues gracefully.

    Args:
        fetcher (DataFetcher): Initialized DataFetcher instance.
        indicators_config (List[IndicatorConfig]): List of indicator configurations.
        logger (logging.Logger): Logger instance for logging.
        rebalancing_date (Optional[str], optional): The date up to which data should be analyzed in "YYYY-MM-DD" format.
                                                  Defaults to today's date if not provided.

    Returns:
        Dict[str, EconomicIndicator]: Dictionary of successfully created EconomicIndicator instances.

    Raises:
        SystemExit: If no EconomicIndicator instances are created successfully.
    """
    economic_data: Dict[str, List[Dict[str, Any]]] = {}
    indicators: Dict[str, EconomicIndicator] = {}

    for indicator_config in indicators_config:
        series_id = indicator_config.series_id
        name = indicator_config.name
        internal_key = indicator_config.internal_key
        try:
            logger.debug(f"Fetching data for {name} (Series ID: {series_id})")
            # Fetch data up to the rebalancing date
            data = fetcher.fetch_data(series_id)
            if not isinstance(data, list) or not data:
                raise ValueError(
                    f"Economic data for {series_id} ({name}) is invalid or empty."
                )
            economic_data[internal_key] = data

            # Create EconomicIndicator instance with the fetched data and rebalancing date
            economic_indicator = EconomicIndicator(
                config=indicator_config,
                data=data,
                rebalancing_date=rebalancing_date,
                logger=logger.getChild("EconomicIndicator"),
            )
            indicators[internal_key] = economic_indicator
            logger.info(f"EconomicIndicator for '{name}' created successfully.")
            logger.debug(f"EconomicIndicator: {economic_indicator}")
        except ValueError as ve:
            logger.warning(f"Data validation issue for {series_id} ('{name}'): {ve}")
            continue  # Continue processing other indicators
        except Exception as e:
            logger.error(
                f"Error processing data for {series_id} ('{name}'): {e}", exc_info=True
            )
            continue  # Continue processing other indicators

    if not indicators:
        logger.critical("No EconomicIndicators were created successfully. Exiting.")
        sys.exit(1)

    return indicators


def initialize_portfolio(logger: logging.Logger) -> Portfolio:
    """
    Initialize the investment portfolio.

    This function creates a new Portfolio instance, logging the initialization process.
    If the portfolio fails to initialize, the application exits gracefully.

    Args:
        logger (logging.Logger): Logger instance for logging.

    Returns:
        Portfolio: Initialized Portfolio instance.

    Raises:
        SystemExit: If initialization of the Portfolio fails.
    """
    try:
        logger.debug("Initializing Portfolio with allocations.")
        portfolio = Portfolio()
        logger.info("Portfolio initialized with initial allocations.")
        return portfolio
    except Exception as e:
        logger.critical(f"Error initializing Portfolio: {e}", exc_info=True)
        sys.exit(1)


def parse_allocations(
    config: Dict[str, Any], logger: logging.Logger
) -> Tuple[
    Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, Dict[str, float]]
]:
    """
    Parse allocations from the configuration.

    Args:
        config (Dict[str, Any]): The loaded configuration dictionary.
        logger (logging.Logger): Logger instance for logging.

    Returns:
        Tuple containing:
            - initial_allocations (Dict[str, float])
            - min_allocations (Dict[str, float])
            - max_allocations (Dict[str, float])
            - allocation_constraints (Dict[str, Dict[str, float]])
    """
    allocations_config = config.get("allocations", {})
    initial_allocations = {}
    min_allocations = {}
    max_allocations = {}
    allocation_constraints = {}

    for asset, alloc_values in allocations_config.items():
        initial = alloc_values.get("initial")
        min_alloc = alloc_values.get("min")
        max_alloc = alloc_values.get("max")

        if initial is None or min_alloc is None or max_alloc is None:
            logger.error(
                f"Allocation for '{asset}' must include 'initial', 'min', and 'max' values."
            )
            sys.exit(1)

        initial_allocations[asset] = initial
        min_allocations[asset] = min_alloc
        max_allocations[asset] = max_alloc

        allocation_constraints[asset] = {"min": min_alloc, "max": max_alloc}

    # Verify that initial allocations sum to 100%
    total_initial_alloc = sum(initial_allocations.values())
    if not math.isclose(total_initial_alloc, 100.0, abs_tol=1e-2):
        logger.error(
            f"Initial allocations sum to {total_initial_alloc}%, expected 100%."
        )
        sys.exit(1)

    return initial_allocations, min_allocations, max_allocations, allocation_constraints
