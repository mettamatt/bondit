# src/data_fetcher.py

"""
data_fetcher.py

This module defines the `DataFetcher` class, responsible for retrieving and managing economic 
data from the Federal Reserve Economic Data (FRED) API. It retrieves data within an adjusted date range 
considering each indicator's specific time frames required. The class ensures data is 
up-to-date by checking release schedules and retrieving new data as necessary, while also caching 
data locally to optimize performance.

Main Functionalities:
- Making API requests to the FRED service.
- Retrieving release IDs and release dates for specific economic data series.
- Checking if cached data is outdated or missing, based on the series' frequency (daily, monthly, quarterly).
- Fetching data from the API for the adjusted date range.
- Storing and managing the retrieved data.

Usage Example:
    from src.data_storage import FredDataStorage
    from src.data_fetcher import DataFetcher

    storage = FredDataStorage(storage_file="./data/fred_data.json")
    api_key = "YOUR_FRED_API_KEY"
    data_fetcher = DataFetcher(storage=storage, api_key=api_key)

    series_id = "GDP"
    data = data_fetcher.fetch_data(series_id)
    print(data)
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, TypeVar, cast

import requests
from dateutil.relativedelta import relativedelta

from .config import ANALYSIS_START_DATE, INDICATORS, IndicatorConfig, IndicatorType
from .data_storage import FredDataStorage, StorageMixin

logger = logging.getLogger("Bondit.DataFetcher")

T = TypeVar("T")


class DataFetcher(StorageMixin):
    """
    Fetches and manages economic data from the FRED API.

    This class handles the retrieval of data, ensuring it is up-to-date based on the release schedule,
    and caches the data locally to optimize performance. It utilizes the `StorageMixin` to manage JSON data
    storage for release dates.

    Enhanced Features:
    - Adjusted date range per indicator based on each indicator's earliest_date and time_frame_weights.
    - Check if the cached data is available and up-to-date.
    - Fetch missing data from the FRED API if the cache doesn't cover the adjusted date range.
    - Update the cache with new data to include the newly fetched date range.
    - Incorporate frequency delay logic to determine when to fetch new data based on expected release schedules.
    """

    BASE_URL: str = "https://api.stlouisfed.org/fred"
    QUARTERLY_DELAY: int = (
        30  # Days to wait after a quarterly release before checking for new data
    )
    MONTHLY_DELAY: int = (
        7  # Days to wait after a monthly release before checking for new data
    )
    DAILY_DELAY: int = (
        1  # Days to wait after a daily release before checking for new data
    )

    FREQUENCY_DELAY_MAP: Dict[str, int] = {
        "DAILY": DAILY_DELAY,
        "MONTHLY": MONTHLY_DELAY,
        "QUARTERLY": QUARTERLY_DELAY,
    }

    def __init__(
        self,
        storage: FredDataStorage,
        api_key: str,
        release_dates_file: str = "./data/fred_release_dates.json",
        logger_instance: Optional[logging.Logger] = None,
        indicators: Optional[List[IndicatorConfig]] = None,
    ) -> None:
        """
        Initialize the DataFetcher.

        Args:
            storage (FredDataStorage): Manages storage of economic data.
            api_key (str): API key for authenticating with the FRED API.
            release_dates_file (str, optional): Path to the JSON file storing release dates.
                Defaults to "./data/fred_release_dates.json".
            logger_instance (Optional[logging.Logger], optional): Logger instance for logging operations.
                If None, a default logger is used.
            indicators (Optional[List[IndicatorConfig]], optional): List of indicator configurations.
                If None, the centralized `INDICATORS` list is used.
        """
        super().__init__(logger=logger_instance)
        self.storage: FredDataStorage = storage
        self.api_key: str = api_key
        self.storage_file: str = release_dates_file
        self.set_storage_file(self.storage_file)
        self.release_dates: Dict[str, str] = cast(
            Dict[str, str], self._load_data(default_data={})
        )
        self.logger: logging.Logger = logger_instance or logging.getLogger(
            "Bondit.DataFetcher"
        )
        self.logger.debug("DataFetcher initialized successfully.")

        # Map each series_id to its corresponding IndicatorConfig
        self.series_config_map: Dict[str, IndicatorConfig] = {
            indicator.series_id: indicator for indicator in indicators or INDICATORS
        }
        self.logger.debug("Series configuration map initialized.")

    def _get_release_date(self, series_id: str) -> Optional[str]:
        """
        Retrieve the stored release date for a specific economic data series.

        Args:
            series_id (str): The ID of the economic data series.

        Returns:
            Optional[str]: The release date in "YYYY-MM-DD" format if available, else None.
        """
        release_date = self.release_dates.get(series_id)
        self.logger.debug(
            f"Retrieved release date for series {series_id}: {release_date}"
        )
        return release_date

    def _get_frequency_delay(self, indicator_type: IndicatorType) -> int:
        """
        Retrieve the frequency delay based on the indicator type.

        Args:
            indicator_type (IndicatorType): The type of the indicator (e.g., 'DAILY', 'MONTHLY', 'QUARTERLY').

        Returns:
            int: The number of days to delay before the next data check.
        """
        delay: int = self.FREQUENCY_DELAY_MAP.get(
            indicator_type.value, self.MONTHLY_DELAY
        )
        self.logger.debug(
            f"Frequency delay for indicator type '{indicator_type.value}': {delay} days."
        )
        return delay

    def _update_release_date(self, series_id: str, new_date: str) -> None:
        """
        Update the stored release date for a series with a delay based on data frequency.

        Args:
            series_id (str): The ID of the economic data series.
            new_date (str): The new release date in "YYYY-MM-DD" format.

        Raises:
            ValueError: If the provided new_date is not in the correct format.
        """
        self.logger.debug(f"Updating release date for series {series_id} to {new_date}")

        try:
            release_date = datetime.strptime(new_date, "%Y-%m-%d")
            self.logger.debug(f"Parsed new release date: {release_date}")
        except ValueError as e:
            self.logger.error(
                f"Invalid date format for series {series_id}: {new_date}. Error: {e}"
            )
            raise

        indicator_config = self.series_config_map.get(series_id)
        frequency_delay = (
            self._get_frequency_delay(indicator_config.indicator_type)
            if indicator_config
            else self.MONTHLY_DELAY
        )

        current_stored_date_str = self.release_dates.get(series_id)
        today = datetime.now()

        if current_stored_date_str:
            try:
                current_stored_date = datetime.strptime(
                    current_stored_date_str, "%Y-%m-%d"
                )
                if today >= current_stored_date:
                    adjusted_date = current_stored_date + timedelta(
                        days=frequency_delay
                    )
                    self.logger.debug(
                        f"Adjusted release date based on current stored date: {adjusted_date}"
                    )
                else:
                    adjusted_date = release_date
                    self.logger.debug(
                        f"Release date not yet reached. Setting adjusted date to: {adjusted_date}"
                    )
            except ValueError as e:
                self.logger.error(
                    f"Invalid stored release date format for series {series_id}: {current_stored_date_str}. Error: {e}"
                )
                adjusted_date = release_date + timedelta(days=frequency_delay)
                self.logger.debug(
                    f"Fallback adjusted date due to error: {adjusted_date}"
                )
        else:
            adjusted_date = release_date + timedelta(days=frequency_delay)
            self.logger.debug(
                f"No current stored date. Setting adjusted date to: {adjusted_date}"
            )

        self.release_dates[series_id] = adjusted_date.strftime("%Y-%m-%d")
        self.logger.debug(
            f"Release date for series {series_id} set to {self.release_dates[series_id]}"
        )

        self._save_data(self.release_dates)
        self.logger.info(
            f"Release date for series {series_id} updated to {adjusted_date.strftime('%Y-%m-%d')}"
        )

    def _make_api_request(
        self, endpoint: str, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Make an API request to the given endpoint with parameters.

        Args:
            endpoint (str): The API endpoint to target.
            params (Dict[str, Any]): Query parameters for the API request.

        Returns:
            Dict[str, Any]: The JSON response from the API.

        Raises:
            requests.RequestException: If the API request fails.
        """
        params["api_key"] = self.api_key
        params["file_type"] = "json"
        url = f"{self.BASE_URL}/{endpoint}"
        self.logger.info(f"Making API request to URL: {url}")
        self.logger.debug(f"Request parameters: {params}")

        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            self.logger.debug(f"API request successful for endpoint {endpoint}.")
            return cast(Dict[str, Any], response.json())
        except requests.HTTPError as e:
            self.logger.error(f"HTTP error during API request: {e}", exc_info=True)
            raise
        except requests.ConnectionError as e:
            self.logger.error(
                f"Connection error during API request: {e}", exc_info=True
            )
            raise
        except requests.Timeout as e:
            self.logger.error(f"Timeout error during API request: {e}", exc_info=True)
            raise
        except requests.RequestException as e:
            self.logger.error(
                f"Request exception during API request: {e}", exc_info=True
            )
            raise

    def _fetch_from_api(
        self, series_id: str, start_date: str, end_date: str
    ) -> List[Dict[str, Any]]:
        """
        Fetch data from the FRED API for the specified series and date range.

        Args:
            series_id (str): The ID of the economic data series to fetch.
            start_date (str): The start date in "YYYY-MM-DD" format.
            end_date (str): The end date in "YYYY-MM-DD" format.

        Returns:
            List[Dict[str, Any]]: A list of observations for the specified series.
        """
        params = {
            "series_id": series_id,
            "observation_start": start_date,
            "observation_end": end_date,
            "sort_order": "asc",  # Use ascending for easier processing
        }
        self.logger.debug(
            f"Fetching data from API for series {series_id} with params {params}."
        )
        try:
            response = self._make_api_request("series/observations", params)
        except requests.RequestException as e:
            self.logger.error(f"API request failed for series {series_id}: {e}")
            raise

        observations = response.get("observations", [])

        if not isinstance(observations, list):
            self.logger.error(
                f"Invalid data format received for series {series_id}: Expected a list, got {type(observations)}."
            )
            return []

        valid_observations: List[Dict[str, Any]] = []
        for obs in observations:
            if isinstance(obs, dict) and "date" in obs and "value" in obs:
                value = obs["value"]
                if value in ("NaN", "."):
                    obs["value"] = None
                else:
                    try:
                        obs["value"] = float(value)
                    except ValueError:
                        self.logger.warning(
                            f"Invalid value format for series {series_id}: {value}. Setting to None."
                        )
                        obs["value"] = None
                valid_observations.append(obs)
            else:
                self.logger.warning(
                    f"Invalid observation format for series {series_id}: {obs}"
                )

        self.logger.info(
            f"Fetched {len(valid_observations)} valid data points for series {series_id} from API."
        )
        return valid_observations

    def _is_data_missing(
        self, data: List[Dict[str, Any]], start_date: datetime, end_date: datetime
    ) -> bool:
        """
        Determine if the cached data is missing or outdated for a specific series and date range.

        Args:
            data (List[Dict[str, Any]]): The cached data for the series.
            start_date (datetime): The requested start date.
            end_date (datetime): The requested end date.

        Returns:
            bool: True if data is missing or outdated, False otherwise.
        """
        if not data:
            self.logger.warning(
                "No cached data available for series. Data is considered missing."
            )
            return True

        try:
            dates_in_cache = [
                datetime.strptime(entry["date"], "%Y-%m-%d") for entry in data
            ]
            earliest_cached_date = min(dates_in_cache)
            latest_cached_date = max(dates_in_cache)
            self.logger.debug(
                f"Cached data range: {earliest_cached_date.strftime('%Y-%m-%d')} to {latest_cached_date.strftime('%Y-%m-%d')}"
            )
        except (KeyError, ValueError) as e:
            self.logger.error(f"Invalid data format in cache. Error: {e}")
            return True

        if earliest_cached_date > start_date or latest_cached_date < end_date:
            self.logger.info(
                "Cached data does not fully cover the required date range."
            )
            return True

        # For simplicity, assume data is up-to-date if cache covers the range
        self.logger.debug("Cached data covers the required date range.")
        return False

    def fetch_data(
        self,
        series_id: str,
    ) -> List[Dict[str, Any]]:
        """
        Fetch economic data for a specified series ID within the adjusted date range.

        Uses cached data if up-to-date and covers the adjusted date range; otherwise, retrieves
        new data from the FRED API.

        This method ensures that new data is retrieved when the cached data isn't fully
        covering the adjusted date range or when new data is expected based on release schedules.

        Args:
            series_id (str): The ID of the economic data series to fetch.

        Returns:
            List[Dict[str, Any]]: A list of data points for the specified series.

        Raises:
            requests.HTTPError: If an HTTP error occurs during API requests.
            Exception: For any unexpected errors during data fetching.
        """
        try:
            self.logger.info(f"Initiating data fetch for series: {series_id}")

            # Retrieve the specific indicator configuration
            indicator_config = self.series_config_map.get(series_id)
            if not indicator_config:
                self.logger.error(
                    f"Indicator configuration not found for series {series_id}."
                )
                return []

            # Compute the required data download start date for this indicator
            earliest_date_dt = datetime.strptime(
                indicator_config.earliest_date, "%Y-%m-%d"
            )
            max_time_frame_years = max(
                tfw.years for tfw in indicator_config.time_frame_weights
            )
            analysis_start_date_dt = datetime.strptime(ANALYSIS_START_DATE, "%Y-%m-%d")
            data_download_start_dt = analysis_start_date_dt - relativedelta(
                years=max_time_frame_years
            )
            # Ensure data_download_start_dt is not earlier than indicator's earliest_date
            data_download_start_dt = max(data_download_start_dt, earliest_date_dt)
            data_download_start_date = data_download_start_dt.strftime("%Y-%m-%d")
            end_date = datetime.now().strftime("%Y-%m-%d")

            self.logger.debug(
                f"Adjusted date range for series {series_id}: {data_download_start_date} to {end_date}"
            )

            start_date_dt = data_download_start_dt
            end_date_dt = datetime.strptime(end_date, "%Y-%m-%d")

            cached_data = self.storage.get_data(series_id)

            # Determine if data is missing or outdated
            if self._is_data_missing(cached_data, start_date_dt, end_date_dt):
                self.logger.info(
                    f"Data is missing or outdated for series {series_id}. Fetching from API."
                )
                if not cached_data:
                    self.logger.info(
                        f"No cached data for series: {series_id}. Fetching data from {data_download_start_date} to {end_date}."
                    )
                    series_data = self._fetch_from_api(
                        series_id, data_download_start_date, end_date
                    )
                    self.storage.replace_data(series_id, series_data)
                else:
                    # Determine the start date for fetching new data
                    latest_cached_date_str = max(
                        entry["date"] for entry in cached_data if entry.get("date")
                    )
                    latest_cached_date_dt = datetime.strptime(
                        latest_cached_date_str, "%Y-%m-%d"
                    )
                    # Ensure we don't set a start date before the required data download start date
                    new_start_date_dt = max(
                        latest_cached_date_dt + timedelta(days=1),
                        data_download_start_dt,
                    )
                    new_start_date = new_start_date_dt.strftime("%Y-%m-%d")

                    self.logger.info(
                        f"Fetching new data for series {series_id} from {new_start_date} to {end_date}."
                    )
                    new_data = self._fetch_from_api(series_id, new_start_date, end_date)

                    if new_data:
                        # Merge new data with cached data, avoiding duplicates
                        combined_data_dict: Dict[str, Dict[str, Any]] = {
                            entry["date"]: entry for entry in cached_data
                        }
                        for entry in new_data:
                            combined_data_dict[entry["date"]] = (
                                entry  # Update or add new entry
                            )

                        # Convert back to a list and sort by date
                        combined_data: List[Dict[str, Any]] = sorted(
                            combined_data_dict.values(), key=lambda x: x["date"]
                        )

                        self.storage.replace_data(series_id, combined_data)
                        self.logger.info(
                            f"Cache updated with {len(new_data)} new records for series: {series_id}"
                        )
                    else:
                        self.logger.warning(
                            f"No new data fetched for series {series_id}. It might be that new data is not yet available."
                        )

                # Update the release date after fetching new data
                new_release_date = datetime.now()
                self._update_release_date(
                    series_id, new_release_date.strftime("%Y-%m-%d")
                )
            else:
                self.logger.info(
                    f"Using cached data for series: {series_id} without fetching new data."
                )

            # Refresh cached_data in case it was updated
            cached_data = self.storage.get_data(series_id)

            if not cached_data:
                self.logger.warning(
                    f"No data available for series {series_id} after attempted fetch."
                )
                return []

            # Filter cached_data to the adjusted date range
            series_data = [
                entry
                for entry in cached_data
                if start_date_dt
                <= datetime.strptime(entry["date"], "%Y-%m-%d")
                <= end_date_dt
            ]
            self.logger.info(
                f"Returning {len(series_data)} records for series {series_id}."
            )
            return series_data

        except requests.HTTPError as e:
            self.logger.error(
                f"HTTP error during data fetch for series {series_id}: {e}",
                exc_info=True,
            )
            raise  # Re-raise the exception after logging

        except Exception as e:
            self.logger.error(
                f"Unexpected error during data fetch for series {series_id}: {e}",
                exc_info=True,
            )
            raise  # Re-raise the exception after logging

    def _get_release_id(self, series_id: str) -> Optional[int]:
        """
        Retrieve the release ID associated with a specific economic data series.

        Args:
            series_id (str): The ID of the economic data series.

        Returns:
            Optional[int]: The release ID if found, else None.
        """
        try:
            params = {"series_id": series_id}
            self.logger.debug(
                f"Fetching release ID for series {series_id} with params {params}."
            )
            response = self._make_api_request("series/release", params)
            releases = response.get("releases", [])

            if isinstance(releases, list) and releases:
                first_release = releases[0]
                release_id_raw = first_release.get("id")
                if release_id_raw is not None:
                    try:
                        release_id = int(release_id_raw)
                        self.logger.debug(
                            f"Found release ID {release_id} for series {series_id}."
                        )
                        return release_id
                    except (ValueError, TypeError):
                        self.logger.warning(
                            f"Release ID for series {series_id} is not a valid integer: {release_id_raw}"
                        )
                else:
                    self.logger.warning(
                        f"First release entry for series {series_id} lacks 'id' key."
                    )
            else:
                self.logger.warning(f"No releases found for series {series_id}.")

        except requests.HTTPError as e:
            self.logger.error(
                f"HTTP error fetching release ID from API for series {series_id}: {e}",
                exc_info=True,
            )
        except Exception as e:
            self.logger.error(
                f"Unexpected error fetching release ID for series {series_id}: {e}",
                exc_info=True,
            )

        return None
