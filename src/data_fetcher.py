# src/data_fetcher.py

"""
data_fetcher.py

This module defines the `DataFetcher` class, responsible for retrieving and managing economic 
data from the Federal Reserve Economic Data (FRED) API. It ensures data is up-to-date 
by checking release schedules and retrieving new data as necessary, while also caching data locally 
to optimize performance.

Main Functionalities:
- Making API requests to the FRED service.
- Retrieving release IDs and release dates for specific economic data series.
- Checking if cached data is outdated or missing, based on the series' frequency (daily, monthly, quarterly).
- Fetching data from the API, focusing on the most recent 6 years.
- Storing and managing the retrieved data.

Note:
This module requires a valid FRED API key to function properly.
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, TypeVar, cast

import requests
from dateutil.relativedelta import relativedelta

from .config import INDICATORS, IndicatorConfig
from .data_storage import FredDataStorage, StorageMixin

logger = logging.getLogger("Bondit.DataFetcher")

T = TypeVar("T")


class DataFetcher(StorageMixin):
    """
    Fetches and manages economic data from the FRED API.

    This class handles the retrieval of data, ensuring it is up-to-date based on the release schedule,
    and caches the data locally to optimize performance. It utilizes the `StorageMixin` to manage JSON data
    storage for release dates.
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
        logger: Optional[logging.Logger] = None,
        indicators: Optional[List[IndicatorConfig]] = None,
    ) -> None:
        """
        Initialize the DataFetcher.

        Args:
            storage (FredDataStorage): Manages storage of economic data.
            api_key (str): API key for authenticating with the FRED API.
            release_dates_file (str, optional): Path to the JSON file storing release dates.
                Defaults to "./data/fred_release_dates.json".
            logger (Optional[logging.Logger], optional): Logger instance for logging operations.
                If None, a default logger is used.
            indicators (Optional[List[IndicatorConfig]], optional): List of indicator configurations.
                If None, the centralized `INDICATORS` list is used.
        """
        super().__init__(logger=logger)
        self.storage: FredDataStorage = storage
        self.api_key: str = api_key
        self.storage_file: str = release_dates_file
        self.set_storage_file(self.storage_file)
        self.release_dates: Dict[str, str] = cast(
            Dict[str, str], self._load_data(default_data={})
        )
        self.logger: logging.Logger = logger or logging.getLogger("Bondit.DataFetcher")
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

    def _get_frequency_delay(self, indicator_type: str) -> int:
        """
        Retrieve the frequency delay based on the indicator type.

        Args:
            indicator_type (str): The type of the indicator (e.g., 'DAILY', 'MONTHLY', 'QUARTERLY').

        Returns:
            int: The number of days to delay before the next data check.
        """
        delay = self.FREQUENCY_DELAY_MAP.get(indicator_type, self.MONTHLY_DELAY)
        self.logger.debug(
            f"Frequency delay for indicator type '{indicator_type}': {delay} days."
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
        if indicator_config:
            frequency_delay = self._get_frequency_delay(indicator_config.indicator_type)
        else:
            self.logger.warning(
                f"No configuration found for series {series_id}. Using default delay."
            )
            frequency_delay = self.MONTHLY_DELAY

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
        except requests.RequestException as e:
            self.logger.error(f"API request failed for endpoint {endpoint}: {e}")
            raise

    def _fetch_from_api(self, series_id: str) -> List[Dict[str, Any]]:
        """
        Fetch the most recent 6 years of data from the FRED API for the specified series.

        Args:
            series_id (str): The ID of the economic data series to fetch.

        Returns:
            List[Dict[str, Any]]: A list of observations for the specified series.
        """
        params = {
            "series_id": series_id,
            "observation_start": (datetime.now() - relativedelta(years=6)).strftime(
                "%Y-%m-%d"
            ),
            "observation_end": datetime.now().strftime("%Y-%m-%d"),
            "sort_order": "desc",
        }
        self.logger.debug(
            f"Fetching data from API for series {series_id} with params {params}."
        )
        try:
            response = self._make_api_request("series/observations", params)
        except Exception as e:
            self.logger.error(f"API request failed for series {series_id}: {e}")
            raise

        observations = response.get("observations", [])

        if not isinstance(observations, list):
            self.logger.error(
                f"Invalid data format received for series {series_id}: Expected a list, got {type(observations)}."
            )
            return []

        valid_observations = []
        for obs in observations:
            if isinstance(obs, dict) and "date" in obs and "value" in obs:
                valid_observations.append(obs)
            else:
                self.logger.warning(
                    f"Invalid observation format for series {series_id}: {obs}"
                )

        self.logger.info(
            f"Fetched {len(valid_observations)} valid data points for series {series_id} from API."
        )
        return valid_observations

    def _is_data_missing(self, data: List[Dict[str, Any]], series_id: str) -> bool:
        """
        Determine if the cached data is missing or outdated for a specific series.

        Args:
            data (List[Dict[str, Any]]): The cached data for the series.
            series_id (str): The ID of the economic data series.

        Returns:
            bool: True if data is missing or outdated, False otherwise.
        """
        if not data:
            self.logger.warning(
                f"No cached data available for series {series_id}. Data is considered missing."
            )
            return True

        try:
            last_date_str = data[-1]["date"]
            last_date = datetime.strptime(last_date_str, "%Y-%m-%d")
            self.logger.debug(
                f"Last cached date for series {series_id}: {last_date.strftime('%Y-%m-%d')}"
            )
        except (KeyError, ValueError) as e:
            self.logger.error(f"Invalid data format for series {series_id}. Error: {e}")
            return True

        today = datetime.now()
        release_date_str = self._get_release_date(series_id)

        if release_date_str:
            try:
                release_date = datetime.strptime(release_date_str, "%Y-%m-%d")
                self.logger.debug(
                    f"Next release date for series {series_id}: {release_date.strftime('%Y-%m-%d')}"
                )
                if today >= release_date > last_date:
                    self.logger.info(
                        f"New data likely available for series {series_id}: Released on {release_date_str}"
                    )
                    return True
                else:
                    self.logger.info(
                        f"No new data needed for series {series_id}: Will check again on {release_date_str}"
                    )
                    return False
            except ValueError as e:
                self.logger.error(
                    f"Invalid release date format for series {series_id}: {release_date_str}. Error: {e}"
                )

        # Fallback logic if no valid release date is available
        indicator_config = self.series_config_map.get(series_id)
        if indicator_config:
            frequency_delay = self._get_frequency_delay(indicator_config.indicator_type)
        else:
            self.logger.warning(
                f"No configuration found for series {series_id}. Using default delay."
            )
            frequency_delay = self.MONTHLY_DELAY

        fallback_date = last_date + timedelta(days=frequency_delay)
        self.logger.debug(
            f"Fallback date for series {series_id}: {fallback_date.strftime('%Y-%m-%d')}"
        )
        if today > fallback_date:
            self.logger.info(
                f"Data for {series_id} is older than expected (Fallback date: {fallback_date.strftime('%Y-%m-%d')}). Refreshing data."
            )
            return True

        self.logger.debug(f"Data for series {series_id} is up-to-date.")
        return False

    def _adjust_release_date(self, series_id: str, fallback_date: datetime) -> None:
        """
        Adjust the release date for a given series.

        Args:
            series_id (str): The ID of the economic data series.
            fallback_date (datetime): The new fallback date to set.
        """
        self.release_dates[series_id] = fallback_date.strftime("%Y-%m-%d")
        self._save_data(self.release_dates)
        self.logger.info(
            f"Adjusted release date for series {series_id} to {fallback_date.strftime('%Y-%m-%d')}."
        )

    def fetch_data(self, series_id: str) -> List[Dict[str, Any]]:
        """
        Fetch economic data for a specified series ID.

        Uses cached data if up-to-date; otherwise, retrieves new data from the FRED API.

        This method ensures that new data is retrieved when the release date has expired.
        If new data is not available after the release date has passed, it adjusts the release date
        to allow for future checks sooner, ensuring that data updates are not missed due to inaccurate release dates.

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
            cached_data = self.storage.get_data(series_id)

            indicator_config = self.series_config_map.get(series_id)
            indicator_type = (
                indicator_config.indicator_type if indicator_config else "MONTHLY"
            )
            frequency_delay = self._get_frequency_delay(indicator_type)

            if cached_data and not self._is_data_missing(cached_data, series_id):
                self.logger.info(f"Using cached data for series: {series_id}")
                return cached_data

            self.logger.info(
                f"Cached data missing or outdated for series: {series_id}. Fetching new data."
            )

            release_id = self._get_release_id(series_id)
            if release_id is not None:
                params = {
                    "release_id": release_id,
                    "limit": 1,
                    "order_by": "release_date",
                    "sort_order": "desc",
                }
                self.logger.debug(
                    f"Fetching release dates for series {series_id} with release_id {release_id}."
                )
                response = self._make_api_request("release/dates", params)
                release_dates = response.get("release_dates", [])
                if isinstance(release_dates, list) and release_dates:
                    new_release_date = release_dates[0].get("date", "")
                    if new_release_date:
                        self._update_release_date(series_id, new_release_date)
                    else:
                        self.logger.warning(
                            f"No valid date found in release_dates for release_id: {release_id}"
                        )
                else:
                    self.logger.warning(
                        f"No release dates found for release_id: {release_id}"
                    )
            else:
                self.logger.warning(
                    f"Could not find release ID for series: {series_id}"
                )

            series_data = self._fetch_from_api(series_id)
            self.logger.debug(
                f"Retrieved {len(series_data)} data points from API for series {series_id}."
            )

            if series_data:
                if cached_data:
                    try:
                        last_cached_date_str = cached_data[-1]["date"]
                        last_cached_date = datetime.strptime(
                            last_cached_date_str, "%Y-%m-%d"
                        )
                        new_entries = [
                            entry
                            for entry in series_data
                            if datetime.strptime(entry["date"], "%Y-%m-%d")
                            > last_cached_date
                        ]
                        if new_entries:
                            self.logger.info(
                                f"New data found for series {series_id}: {len(new_entries)} new records."
                            )
                            self.storage.replace_data(series_id, series_data)
                            self.logger.info(
                                f"Data for series {series_id} saved successfully."
                            )
                        else:
                            self.logger.info(
                                f"No new data found for series {series_id}. Adjusting release date for future checks."
                            )
                            fallback_date = last_cached_date + timedelta(
                                days=frequency_delay
                            )
                            self._adjust_release_date(series_id, fallback_date)
                    except (KeyError, ValueError) as e:
                        self.logger.error(
                            f"Error processing cached data for series {series_id}: {e}"
                        )
                        fallback_date = datetime.now() + timedelta(days=frequency_delay)
                        self._adjust_release_date(series_id, fallback_date)
                else:
                    self.storage.replace_data(series_id, series_data)
                    self.logger.info(f"Data for series {series_id} saved successfully.")
            else:
                self.logger.warning(
                    f"No data points found for series {series_id} from API."
                )

                if cached_data:
                    try:
                        last_date_str = cached_data[-1]["date"]
                        last_date = datetime.strptime(last_date_str, "%Y-%m-%d")
                        fallback_date = last_date + timedelta(days=frequency_delay)
                        self._adjust_release_date(series_id, fallback_date)
                    except (KeyError, ValueError) as e:
                        self.logger.error(
                            f"Failed to adjust release date for series {series_id}: {e}"
                        )

            return series_data

        except Exception as e:
            self.logger.error(
                f"Unexpected error fetching data for series {series_id}: {e}"
            )
            raise

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
                f"HTTP error fetching release ID from API for series {series_id}: {e}"
            )
        except Exception as e:
            self.logger.error(
                f"Unexpected error fetching release ID for series {series_id}: {e}"
            )

        return None
