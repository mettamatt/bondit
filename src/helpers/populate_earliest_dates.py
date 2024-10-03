# src/populate_earliest_dates.py

"""
populate_earliest_dates.py

This script populates the `earliest_date` field for each economic indicator in the 
`src/config.py` file by fetching data from the Federal Reserve Economic Data (FRED) API.
"""

import logging
import os
import sys
from typing import List, TypedDict, cast

import requests

from src.config import INDICATORS, IndicatorConfig
from src.utils import initialize_logger, load_configuration


class SeriesData(TypedDict, total=False):
    observation_start: str
    # Add other relevant fields if necessary


class SeriesResponse(TypedDict, total=False):
    series: List[SeriesData]
    # Add other relevant fields if necessary


def fetch_earliest_date(series_id: str, api_key: str, logger: logging.Logger) -> str:
    """
    Fetch the earliest available date for a given series_id from the FRED API.

    Args:
        series_id (str): The FRED series ID.
        api_key (str): Your FRED API key.
        logger (logging.Logger): Logger instance for logging.

    Returns:
        str: The earliest available date in "YYYY-MM-DD" format.

    Raises:
        Exception: If the API request fails or the response is invalid.
    """
    endpoint = "https://api.stlouisfed.org/fred/series"
    params = {"series_id": series_id, "api_key": api_key, "file_type": "json"}

    try:
        response = requests.get(endpoint, params=params)
        response.raise_for_status()
        data = cast(SeriesResponse, response.json())
        # Corrected the response parsing for the "series" key
        series_list = data.get("series", [])
        if not series_list:
            raise ValueError(f"No series data found for series_id: {series_id}")

        observation_start = series_list[0].get("observation_start")
        if isinstance(observation_start, str):
            logger.debug(f"Series ID: {series_id} | Earliest Date: {observation_start}")
            return observation_start
        else:
            raise ValueError(
                f"'observation_start' is not a string for series_id: {series_id}"
            )
    except Exception as e:
        raise Exception(f"Failed to fetch earliest date for {series_id}: {e}")


def backup_config_file(
    config_path: str, backup_path: str, logger: logging.Logger
) -> None:
    """
    Create a backup of the existing config.py file.

    Args:
        config_path (str): Path to the original config.py.
        backup_path (str): Path to save the backup config.py.
        logger (logging.Logger): Logger instance for logging.
    """
    try:
        with open(config_path, "r") as original, open(backup_path, "w") as backup:
            backup.write(original.read())
        logger.info(f"Backup of config.py created at {backup_path}")
    except Exception as e:
        logger.critical(f"Failed to create backup of config.py: {e}")
        sys.exit(1)


def update_config_file(
    config_path: str, indicators: List[IndicatorConfig], logger: logging.Logger
) -> None:
    """
    Update the config.py file with the fetched earliest_date values.

    Args:
        config_path (str): Path to the config.py file.
        indicators (List[IndicatorConfig]): List of IndicatorConfig instances with updated earliest_date.
        logger (logging.Logger): Logger instance for logging.
    """
    try:
        with open(config_path, "r") as file:
            lines = file.readlines()

        with open(config_path, "w") as file:
            for line in lines:
                # Identify lines corresponding to INDICATORS list
                if "IndicatorConfig(" in line and "earliest_date" not in line:
                    # Find the corresponding IndicatorConfig
                    for indicator in indicators:
                        if indicator.series_id in line:
                            # Modify the line to include earliest_date
                            if line.strip().endswith(","):
                                line = (
                                    line.rstrip(",\n")
                                    + f',\n            earliest_date="{indicator.earliest_date}",\n        ),\n'
                                )
                            break
                file.write(line)

        logger.info("config.py has been updated with earliest_date values.")
    except Exception as e:
        logger.critical(f"Failed to update config.py: {e}")
        sys.exit(1)


def main() -> None:
    # Initialize logger
    logger = initialize_logger("PopulateEarliestDates")

    # Define the fixed configuration file path
    config_path = os.path.join(os.path.dirname(__file__), "config", "config.yaml")
    logger.debug(f"Configuration path set to: {config_path}")

    # Define the required configuration fields
    required_config_fields = [
        "storage.fred_file_path",
        "api.fred_api_key",
    ]

    # Load and validate the configuration settings
    config = load_configuration(config_path, logger, required_config_fields)

    # Extract API_KEY and config file paths from the loaded configuration
    API_KEY = config["api"]["fred_api_key"]
    CONFIG_FILE_PATH = os.path.join(os.path.dirname(__file__), "src", "config.py")
    BACKUP_CONFIG_FILE_PATH = os.path.join(
        os.path.dirname(__file__), "src", "config_backup.py"
    )

    if not API_KEY or API_KEY == "YOUR_FRED_API_KEY":
        logger.critical(
            "Please set your FRED API key in the configuration file before running."
        )
        sys.exit(1)

    # Backup the existing config.py
    backup_config_file(CONFIG_FILE_PATH, BACKUP_CONFIG_FILE_PATH, logger)

    updated_indicators: List[IndicatorConfig] = []
    for indicator in INDICATORS:
        try:
            earliest_date = fetch_earliest_date(indicator.series_id, API_KEY, logger)
            updated_indicator = IndicatorConfig(
                series_id=indicator.series_id,
                name=indicator.name,
                internal_key=indicator.internal_key,
                description=indicator.description,
                indicator_type=indicator.indicator_type,
                time_frame_weights=indicator.time_frame_weights,
                calculation_method=indicator.calculation_method,
                thresholds=indicator.thresholds,
                rule_weight=indicator.rule_weight,
                earliest_date=earliest_date,
            )
            updated_indicators.append(updated_indicator)
        except Exception as e:
            logger.error(e)
            logger.warning(f"Skipping series_id: {indicator.series_id}")
            continue

    # Update the config.py file with earliest_date values
    update_config_file(CONFIG_FILE_PATH, updated_indicators, logger)

    logger.info("Earliest dates have been populated in config.py.")


if __name__ == "__main__":
    main()
