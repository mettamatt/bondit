# src/data_storage.py

"""
data_storage.py

This module manages the storage and retrieval of economic data from the FRED API and investment recommendations.
It includes classes for handling JSON data storage, ensuring that data is consistently saved, loaded, and updated.
The module is designed with simplicity and efficiency in mind, leveraging mixins for shared functionality.

Classes:
    StorageMixin: Provides shared methods for loading and saving JSON data with error handling.
    FredDataStorage: Manages storage of economic data retrieved from the FRED API, replacing series data when updated.
    RecommendationStorage: Manages storage and retrieval of investment recommendations, with functionality to add new recommendations.
"""

import json
import logging
import os
from datetime import datetime, timezone
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


class RecommendationStorage(StorageMixin):
    """
    Manages storage and retrieval of investment recommendations.

    Provides functionality to add new recommendations, retrieve them,
    and filter them based on dates. Ensures that duplicate recommendations
    are not stored multiple times.
    """

    def __init__(self, storage_file: str, logger: Optional[logging.Logger] = None):
        """
        Initialize the RecommendationStorage with the path to the storage file.

        Args:
            storage_file (str): The path to the JSON file where recommendations are stored.
            logger (Optional[logging.Logger]): Logger instance for logging operations.
                                               If None, a default logger is used.
        """
        super().__init__(logger=logger)
        self.storage_file = storage_file
        self.set_storage_file(storage_file)

    def add_recommendation(
        self,
        scenario: str,
        allocation: Dict[str, float],
        economic_indicators: Dict[str, float],
    ) -> None:
        """
        Add a new investment recommendation to the storage if it does not already exist.

        Args:
            scenario (str): The investment scenario being recommended.
            allocation (Dict[str, float]): The recommended asset allocation.
            economic_indicators (Dict[str, float]): Relevant economic indicators.

        Raises:
            ValueError: If the recommendation format is invalid.
        """
        new_recommendation = self._create_recommendation(
            scenario, allocation, economic_indicators
        )

        # Load existing recommendations
        recommendations: List[Dict[str, Any]] = self._load_recommendations()

        # Check for duplicate recommendation
        if self._is_duplicate(recommendations, new_recommendation):
            self.logger.info("Duplicate recommendation found. Not saving.")
            return

        # Add the new recommendation and save
        recommendations.append(new_recommendation)
        self._save_recommendations(recommendations)
        self.logger.info("Recommendation added successfully.")

    def get_recommendations(
        self, start_date: Optional[str] = None, end_date: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve a list of investment recommendations, optionally filtered by date range.

        Args:
            start_date (str, optional): The start date for filtering recommendations (ISO format).
            end_date (str, optional): The end date for filtering recommendations (ISO format).

        Returns:
            List[Dict[str, Any]]: A list of recommendations. Each recommendation is a dictionary
                containing the stored data (timestamp, scenario, allocation, economic indicators).

        Raises:
            ValueError: If the provided date formats are invalid.
        """
        recommendations: List[Dict[str, Any]] = self._load_recommendations()

        if not recommendations:
            self.logger.info("No recommendations found in storage.")
            return []

        try:
            if start_date:
                start_datetime = datetime.fromisoformat(start_date)
                recommendations = [
                    r
                    for r in recommendations
                    if datetime.fromisoformat(r["timestamp"]) >= start_datetime
                ]

            if end_date:
                end_datetime = datetime.fromisoformat(end_date)
                recommendations = [
                    r
                    for r in recommendations
                    if datetime.fromisoformat(r["timestamp"]) <= end_datetime
                ]
        except ValueError as e:
            self.logger.error(f"Invalid date format: {e}")
            raise ValueError("Invalid date format. Please use ISO format (YYYY-MM-DD).")

        self.logger.debug(
            f"Retrieved {len(recommendations)} recommendations after filtering."
        )
        return recommendations

    def get_latest_recommendation(self) -> Optional[Dict[str, Any]]:
        """
        Retrieve the most recent recommendation from the storage.

        Returns:
            Optional[Dict[str, Any]]: The most recent recommendation, or None if no recommendations exist.
                                        The dictionary contains the stored recommendation data (timestamp, scenario,
                                        allocation, economic indicators).
        """
        recommendations: List[Dict[str, Any]] = self.get_recommendations()

        if not recommendations:
            self.logger.info("No recommendations found. Returning None.")
            return None

        # Find the recommendation with the latest timestamp
        latest_recommendation = max(
            recommendations, key=lambda x: datetime.fromisoformat(x["timestamp"])
        )
        self.logger.debug(f"Latest recommendation retrieved: {latest_recommendation}")
        return latest_recommendation

    def _create_recommendation(
        self,
        scenario: str,
        allocation: Dict[str, float],
        economic_indicators: Dict[str, float],
    ) -> Dict[str, Any]:
        """
        Create a recommendation dictionary with the current timestamp.

        Args:
            scenario (str): The investment scenario being recommended.
            allocation (Dict[str, float]): The recommended asset allocation.
            economic_indicators (Dict[str, float]): Relevant economic indicators.

        Returns:
            Dict[str, Any]: A dictionary representing the recommendation.
        """
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "scenario": scenario,
            "allocation": allocation,
            "economic_indicators": economic_indicators,
        }

    def _load_recommendations(self) -> List[Dict[str, Any]]:
        """
        Load the list of recommendations from storage.

        Returns:
            List[Dict[str, Any]]: A list of existing recommendations.
        """
        data: Dict[str, Any] = self._load_data(default_data={"recommendations": []})
        recommendations = cast(List[Dict[str, Any]], data.get("recommendations", []))
        self.logger.debug(
            f"Loaded {len(recommendations)} recommendations from storage."
        )
        return recommendations

    def _save_recommendations(self, recommendations: List[Dict[str, Any]]) -> None:
        """
        Save the list of recommendations to storage.

        Args:
            recommendations (List[Dict[str, Any]]): The list of recommendations to save.
        """
        self._save_data({"recommendations": recommendations})
        self.logger.debug(f"Saved {len(recommendations)} recommendations to storage.")

    def _is_duplicate(
        self, recommendations: List[Dict[str, Any]], new_recommendation: Dict[str, Any]
    ) -> bool:
        """
        Check if the new recommendation is a duplicate of an existing one.

        Args:
            recommendations (List[Dict[str, Any]]): The list of existing recommendations.
            new_recommendation (Dict[str, Any]): The new recommendation to check.

        Returns:
            bool: True if duplicate exists, False otherwise.
        """
        is_duplicate = any(
            r["scenario"] == new_recommendation["scenario"]
            and r["allocation"] == new_recommendation["allocation"]
            and r["economic_indicators"] == new_recommendation["economic_indicators"]
            for r in recommendations
        )
        if is_duplicate:
            self.logger.debug("Duplicate recommendation detected.")
        else:
            self.logger.debug("No duplicate recommendation found.")
        return is_duplicate
