import requests
import pandas as pd
from tqdm import tqdm
from datetime import datetime, timedelta
import os
import json
import time
import logging

# --------------------------- Configuration ---------------------------

# List of protocols to fetch. Add more protocols as needed.
PROTOCOLS = [
    "conductivities",
    "dissolved_oxygens",
    "hydrology_alkalinities",
    "hydrology_phs",
    "nitrates",
    "salinities",
    "surface_temperature_noons",
    "surface_temperatures",
    "transparencies",
    "water_temperatures"
    # Add more protocols here
]

# API endpoint template
API_URL_TEMPLATE = "https://api.globe.gov/search/v1/measurement/protocol/measureddate/"

# Default date range for data fetching
START_DATE = "1990-01-01"
END_DATE = "2025-01-11"

# API request parameters
GEOJSON = "TRUE"
SAMPLE = "FALSE"

# HTTP headers for the request
HEADERS = {
    "accept": "application/stream+json"
}

# Directory to save CSV files and logs
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# Log file configuration
LOG_FILE = os.path.join(DATA_DIR, "data_fetch.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)

# Constants for batching and retries
MAX_RECORDS_PER_REQUEST = 1_000_000  # Maximum records per API request
MAX_RETRIES = 5                       # Maximum number of retries for failed requests
RETRY_WAIT_SECONDS = 20               # Wait time between retries (in seconds)

# --------------------------- Utility Functions ---------------------------

def split_date_range(start_date: str, end_date: str):
    """
    Splits a given date range into two approximately equal halves.

    Args:
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.

    Returns:
        tuple: Two tuples representing the split date ranges.
    """
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    delta = end_dt - start_dt
    mid_dt = start_dt + delta / 2
    mid_str = mid_dt.strftime("%Y-%m-%d")
    return (start_date, mid_str), (mid_str + "T00:00:00", end_date)

def sanitize_protocol(protocol: str):
    """
    Sanitizes the protocol name by removing underscores to match JSON property keys.

    Args:
        protocol (str): The original protocol name.

    Returns:
        str: Sanitized protocol name.
    """
    return protocol.replace("_", "")

def construct_csv_filename(protocol: str, min_date: str, max_date: str):
    """
    Constructs the CSV filename based on protocol and date range.

    Args:
        protocol (str): The protocol name.
        min_date (str): Minimum date in 'YYYY-MM-DD' format.
        max_date (str): Maximum date in 'YYYY-MM-DD' format.

    Returns:
        str: Formatted CSV filename.
    """
    return f"{protocol}_{min_date}_{max_date}.csv"

def construct_curl_command(api_url: str, headers: dict):
    """
    Constructs the equivalent curl command for the given API request.

    Args:
        api_url (str): The full API URL with query parameters.
        headers (dict): Dictionary of HTTP headers.

    Returns:
        str: The equivalent curl command as a string.
    """
    header_str = " ".join([f"-H \"{key}: {value}\"" for key, value in headers.items()])
    curl_cmd = f"curl -X GET \"{api_url}\" {header_str}"
    return curl_cmd

def check_existing_csv(protocol: str):
    """
    Checks if a CSV file for the given protocol already exists.

    Args:
        protocol (str): The protocol name.

    Returns:
        bool: True if CSV exists, False otherwise.
    """
    existing_files = [f for f in os.listdir(DATA_DIR) if f.startswith(f"{protocol}_") and f.endswith(".csv")]
    return len(existing_files) > 0

# --------------------------- Core Functions ---------------------------

def fetch_batch(protocol: str, start_date: str, end_date: str, retries: int = 0):
    """
    Fetches a batch of data for a specific protocol and date range.

    Implements retry logic with exponential backoff for handling throttling and server errors.

    Args:
        protocol (str): The protocol name.
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.
        retries (int, optional): Current retry attempt. Defaults to 0.

    Returns:
        list or None: List of property dictionaries if successful, else None.
    """
    params = {
        "protocols": protocol,
        "startdate": start_date,
        "enddate": end_date,
        "geojson": GEOJSON,
        "sample": SAMPLE
    }

    # Construct the full API URL with query parameters
    api_url = f"{API_URL_TEMPLATE}?protocols={protocol}&startdate={start_date}&enddate={end_date}&geojson={GEOJSON}&sample={SAMPLE}"

    # Construct the equivalent curl command
    curl_command = construct_curl_command(api_url, HEADERS)
    logging.info(f"Executing CURL Command: {curl_command}")

    try:
        # Make the GET request with streaming enabled
        response = requests.get(api_url, headers=HEADERS, stream=True, timeout=120)
        logging.info(f"Fetching protocol '{protocol}' from {start_date} to {end_date}. Attempt {retries + 1}.")

        # Handle rate limiting (HTTP 429)
        if response.status_code == 429:
            if retries < MAX_RETRIES:
                wait_time = RETRY_WAIT_SECONDS * (2 ** retries)  # Exponential backoff
                logging.warning(f"Throttled! Retrying after {wait_time} seconds... (Retry {retries + 1}/{MAX_RETRIES})")
                time.sleep(wait_time)
                return fetch_batch(protocol, start_date, end_date, retries + 1)
            else:
                logging.error(f"Max retries reached for protocol '{protocol}' between {start_date} and {end_date}. Skipping this batch.")
                return None

        # Handle server errors (HTTP 5xx)
        elif 500 <= response.status_code < 600:
            if retries < MAX_RETRIES:
                wait_time = RETRY_WAIT_SECONDS * (2 ** retries)
                logging.warning(f"Server error ({response.status_code})! Retrying after {wait_time} seconds... (Retry {retries + 1}/{MAX_RETRIES})")
                time.sleep(wait_time)
                return fetch_batch(protocol, start_date, end_date, retries + 1)
            else:
                logging.error(f"Max retries reached for protocol '{protocol}' between {start_date} and {end_date}. Skipping this batch.")
                return None

        # Raise exception for other HTTP errors
        response.raise_for_status()

        # Parse the JSON response
        data = response.json()
        features = data.get('features', [])

        if not features:
            logging.warning(f"No features found for protocol '{protocol}' between {start_date} and {end_date}.")
            return []

        # Dynamically parse features into a list of records
        records = []
        for feature in features:
            properties = feature.get('properties', {})
            geometry = feature.get('geometry', {})
            coordinates = geometry.get('coordinates', [])

            # Flatten properties and geometry into a single dictionary
            record = {**properties,
                      "latitude": coordinates[1] if len(coordinates) > 1 else None,
                      "longitude": coordinates[0] if len(coordinates) > 0 else None}
            records.append(record)

        logging.info(f"Fetched {len(records)} records for protocol '{protocol}' from {start_date} to {end_date}.")
        return records

    except requests.exceptions.RequestException as e:
        if retries < MAX_RETRIES:
            wait_time = RETRY_WAIT_SECONDS * (2 ** retries)
            logging.warning(f"Error fetching protocol '{protocol}' between {start_date} and {end_date}: {e}. Retrying after {wait_time} seconds... (Retry {retries + 1}/{MAX_RETRIES})")
            time.sleep(wait_time)
            return fetch_batch(protocol, start_date, end_date, retries + 1)
        else:
            logging.error(f"Max retries reached for protocol '{protocol}' between {start_date} and {end_date}. Skipping this batch.")
            return None

def fetch_protocol_data(protocol: str):
    """
    Fetches all data for a given protocol by handling batching based on record counts.

    Splits the date range into smaller batches if the number of records exceeds the API limit.

    Args:
        protocol (str): The protocol name.

    Returns:
        dict: Summary containing protocol name, number of records, min date, and max date.
    """
    all_data = []            # Aggregated data for the protocol
    protocol_min_date = None  # Minimum 'MeasuredAt' date
    protocol_max_date = None  # Maximum 'MeasuredAt' date
    batches = [(START_DATE, END_DATE)]  # Initial date range batch
    failed_batches = []     # List to keep track of failed batches

    # Initialize a progress bar for the current protocol
    with tqdm(total=1, desc=f"Fetching {protocol}", position=0, leave=True) as pbar_protocol:
        while batches:
            current_start, current_end = batches.pop(0)
            records = fetch_batch(protocol, current_start, current_end)

            if records is None:
                # If the batch failed after retries, log it
                failed_batches.append((current_start, current_end))
                pbar_protocol.update(1)
                continue

            if len(records) >= MAX_RECORDS_PER_REQUEST:
                # If the batch size exceeds the maximum limit, split the date range
                (split_start, split_mid), (split_mid_plus, split_end) = split_date_range(current_start, current_end)
                batches.append((split_start, split_mid))
                batches.append((split_mid_plus, split_end))
                pbar_protocol.total += 1  # Update total for the new batches
                pbar_protocol.refresh()
                logging.info(f"Batch size exceeded {MAX_RECORDS_PER_REQUEST}. Splitting into two batches.")
            else:
                # Update min and max dates based on the fetched data
                for record in records:
                    measured_at = record.get("MeasuredAt")
                    if measured_at:
                        if not protocol_min_date or measured_at < protocol_min_date:
                            protocol_min_date = measured_at
                        if not protocol_max_date or measured_at > protocol_max_date:
                            protocol_max_date = measured_at

                # Append records
                all_data.extend(records)
                logging.info(f"Processed batch: {len(records)} records.")

                pbar_protocol.update(1)

    if failed_batches:
        logging.warning(f"Failed to fetch the following batches for protocol '{protocol}':")
        for fb_start, fb_end in failed_batches:
            logging.warning(f"  Start: {fb_start}, End: {fb_end}")

    if not all_data:
        logging.info(f"No data found for protocol '{protocol}'.")
        return {
            "protocol": protocol,
            "records": 0,
            "min_date": None,
            "max_date": None
        }

    # Convert aggregated data to a pandas DataFrame with dynamic flattening
    df = pd.json_normalize(all_data)

    # Ensure 'MeasuredAt' is in datetime format
    measured_at_column = f"{sanitize_protocol(protocol)}MeasuredAt"
    if measured_at_column in df.columns:
        df[measured_at_column] = pd.to_datetime(df[measured_at_column], errors='coerce')

    # If min_date or max_date wasn't set during fetching, derive from the DataFrame
    if not protocol_min_date or not protocol_max_date:
        if measured_at_column in df.columns:
            protocol_min_date = df[measured_at_column].min()
            protocol_max_date = df[measured_at_column].max()

    # Format dates for the CSV filename
    min_date_str = protocol_min_date.strftime("%Y-%m-%d") if protocol_min_date else "min_date"
    max_date_str = protocol_max_date.strftime("%Y-%m-%d") if protocol_max_date else "max_date"

    # Construct the CSV filename
    csv_filename = construct_csv_filename(protocol, min_date_str, max_date_str)
    csv_path = os.path.join(DATA_DIR, csv_filename)

    # Save the DataFrame to CSV
    try:
        df.to_csv(csv_path, index=False)
        logging.info(f"Saved data for protocol '{protocol}' to '{csv_path}'.")
    except Exception as e:
        logging.error(f"Failed to save CSV for protocol '{protocol}': {e}")

    return {
        "protocol": protocol,
        "records": len(df),
        "min_date": protocol_min_date.strftime("%Y-%m-%d %H:%M:%S") if protocol_min_date else None,
        "max_date": protocol_max_date.strftime("%Y-%m-%d %H:%M:%S") if protocol_max_date else None
    }

# --------------------------- Main Execution ---------------------------

def main():
    """
    Main function to orchestrate the data fetching process for all protocols.

    - Checks if the data already exists to avoid redundant fetching.
    - Fetches and processes data for each protocol.
    - Handles batching for large datasets.
    - Logs all activities.
    """
    summary_records = []

    for protocol in PROTOCOLS:
        # Check if CSV already exists for the protocol
        if check_existing_csv(protocol):
            logging.info(f"CSV for protocol '{protocol}' already exists. Skipping fetch.")
            # Optionally, append existing file details to summary
            existing_files = [f for f in os.listdir(DATA_DIR) if f.startswith(f"{protocol}_") and f.endswith(".csv")]
            for file in existing_files:
                try:
                    # Extract min and max dates from filename
                    parts = file.rstrip('.csv').split('_')
                    if len(parts) >= 3:
                        min_date_str = parts[-2]
                        max_date_str = parts[-1]
                        records = pd.read_csv(os.path.join(DATA_DIR, file)).shape[0]
                        summary_records.append({
                            "protocol": protocol,
                            "records": records,
                            "min_date": min_date_str,
                            "max_date": max_date_str
                        })
                except Exception as e:
                    logging.error(f"Error processing existing file '{file}': {e}")
            continue  # Move to the next protocol

        # Fetch data for the protocol
        summary = fetch_protocol_data(protocol)
        summary_records.append(summary)

    # Create a summary DataFrame from the results
    summary_df = pd.DataFrame(summary_records)

    # Display the summary
    logging.info("\nSummary of Fetched Protocols:")
    logging.info(f"\n{summary_df}")

    # Save the summary to a CSV file
    summary_csv = os.path.join(DATA_DIR, "summary.csv")
    try:
        summary_df.to_csv(summary_csv, index=False)
        logging.info(f"Summary saved to '{summary_csv}'.")
    except Exception as e:
        logging.error(f"Failed to save summary CSV: {e}")

# --------------------------- Entry Point ---------------------------

if __name__ == "__main__":
    main()