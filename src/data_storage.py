# src/data_storage.py

"""
data_storage.py

This module manages the storage and retrieval of economic data from the FRED API and investment recommendations.
It includes classes for handling JSON data storage, ensuring that data is consistently saved, loaded, and updated.
The module is designed with simplicity and efficiency in mind, leveraging mixins for shared functionality.

Classes:
    StorageMixin: Provides shared methods for loading and saving JSON data with error handling.
    FredDataStorage: Manages storage of economic data retrieved from the FRED API, replacing series data when updated.
"""

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, TypeVar, cast

T = TypeVar("T")


class StorageMixin:
    """
    Mixin class that provides shared methods for loading and saving JSON data.

    This mixin includes error handling to manage issues with file I/O and JSON parsing, ensuring data integrity.
    It serves as a foundation for other storage-related classes to inherit common functionalities.
    """

    storage_file: Optional[str] = None

    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the StorageMixin with an optional logger.

        Args:
            logger (Optional[logging.Logger]): Logger instance to use for logging.
                                               If None, a default logger is used.
        """
        self.logger = logger or logging.getLogger("Bondit.DataStorage")

    def set_storage_file(self, path: str) -> None:
        """
        Set the path to the storage file.

        Args:
            path (str): The file path where data will be stored.
        """
        self.storage_file = path
        self.logger.debug(f"Storage file set to {path}.")

    def _load_data(self, default_data: Optional[T] = None) -> T:
        """
        Load data from the storage file. If the file does not exist or is corrupted, return default_data.

        Args:
            default_data (Optional[T]): The data to return if loading fails or file is not found.
                                        Defaults to None.

        Returns:
            T: The data loaded from the JSON file or default_data if an error occurs.

        Raises:
            ValueError: If the storage file path is not set.
        """
        if self.storage_file is None:
            raise ValueError("Storage file path is not set.")

        if not os.path.exists(self.storage_file):
            self.logger.info(
                f"Storage file {self.storage_file} does not exist. Using default data."
            )
            if default_data is not None:
                return default_data
            else:
                # Default to empty dictionary if no default data is provided
                return cast(T, {})

        try:
            with open(self.storage_file, "r") as file:
                self.logger.debug(f"Loading data from {self.storage_file}.")
                return cast(T, json.load(file))
        except (IOError, json.JSONDecodeError) as e:
            self.logger.error(f"Error loading data from {self.storage_file}: {e}")
            if default_data is not None:
                return default_data
            else:
                return cast(T, {})

    def _save_data(self, data: T) -> None:
        """
        Save data to the storage file, ensuring persistence of JSON data.

        Args:
            data (T): The data to save to the JSON file.

        Raises:
            ValueError: If the storage file path is not set.
        """
        if self.storage_file is None:
            raise ValueError("Storage file path is not set.")

        try:
            with open(self.storage_file, "w") as file:
                json.dump(data, file, indent=4)
                self.logger.debug(f"Data successfully saved to {self.storage_file}.")
        except IOError as e:
            self.logger.error(f"Error saving data to {self.storage_file}: {e}")


class FredDataStorage(StorageMixin):
    """
    Manages the storage and retrieval of economic data from the FRED API.

    This class replaces the entire dataset for a specific series when new data is retrieved,
    ensuring that the stored data remains current and accurate.
    """

    def __init__(self, storage_file: str, logger: Optional[logging.Logger] = None):
        """
        Initialize the FredDataStorage with the path to the storage file.

        Args:
            storage_file (str): The path to the JSON file where data is stored.
            logger (Optional[logging.Logger]): Logger instance for logging operations.
                                               If None, a default logger is used.
        """
        super().__init__(logger=logger)
        self.logger = logger or logging.getLogger("Bondit.DataStorage.FredDataStorage")
        self.storage_file = storage_file
        self.set_storage_file(storage_file)

    def get_data(self, series_id: str) -> List[Dict[str, Any]]:
        """
        Retrieve stored data for a specific economic series.

        Args:
            series_id (str): The ID of the economic data series to retrieve.

        Returns:
            List[Dict[str, Any]]: A list of data points for the specified series.
        """
        data: Dict[str, Any] = self._load_data(default_data={})
        series_data = cast(List[Dict[str, Any]], data.get(series_id, []))
        self.logger.debug(
            f"Retrieved data for series_id {series_id}: {len(series_data)} records found."
        )
        return series_data

    def replace_data(self, series_id: str, new_data: List[Dict[str, Any]]) -> None:
        """
        Replace the existing dataset for a specific series with new data after validation.

        Args:
            series_id (str): The ID of the economic data series.
            new_data (List[Dict[str, Any]]): A list of new data points to replace the existing data.

        Raises:
            ValueError: If the new_data does not conform to the expected structure.
        """
        if not self._validate_new_data(new_data):
            self.logger.error(
                f"Validation failed for new_data of series {series_id}. Data not replaced."
            )
            raise ValueError(f"Invalid data format for series {series_id}.")

        data: Dict[str, Any] = self._load_data(default_data={})
        data[series_id] = new_data  # Safely assign new data to the series
        self._save_data(data)
        self.logger.info(
            f"Data for series {series_id} has been successfully replaced with {len(new_data)} new records."
        )

    def _validate_new_data(self, new_data: List[Dict[str, Any]]) -> bool:
        """
        Validate the structure and contents of the new_data to ensure consistency.

        Args:
            new_data (List[Dict[str, Any]]): The new data to validate.

        Returns:
            bool: True if validation passes, False otherwise.
        """
        required_keys = {"date", "value"}
        for entry in new_data:
            if not isinstance(entry, dict):
                self.logger.error("Data entry is not a dictionary.")
                return False
            if not required_keys.issubset(entry.keys()):
                self.logger.error(
                    f"Data entry missing required keys: {required_keys}. Entry: {entry}"
                )
                return False
            # Validate date format
            if not isinstance(entry["date"], str):
                self.logger.error(
                    f"Invalid type for date: {entry['date']} (expected str)."
                )
                return False
            try:
                datetime.strptime(entry["date"], "%Y-%m-%d")
            except ValueError:
                self.logger.error(
                    f"Invalid date format for entry: {entry['date']}. Expected 'YYYY-MM-DD'."
                )
                return False
            # Validate value type
            if not isinstance(entry["value"], (str, float, int)):
                self.logger.error(
                    f"Invalid type for value: {entry['value']} (expected str, float, or int)."
                )
                return False
        self.logger.debug("New data passed validation checks.")
        return True
