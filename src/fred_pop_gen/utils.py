from functools import reduce

import numpy as np
import pandas as pd
import requests

from fred_pop_gen.config import CENSUS_YEAR, STATE_FIPS, DATA

COUNTIES_FILE = DATA / f"input/counties-{STATE_FIPS}.txt"


def _download_county_fips() -> None:
    """
    Downloads county FIPS codes from the census API and writes them to
    `COUNTIES_FILE`.
    """
    # NOTE: we are using a DUMMY variable (B01001_001E) in the API call
    # all we really need is the state and county values
    url = f"https://api.census.gov/data/{CENSUS_YEAR}/acs/acs5?get=B01001_001E&for=county:*&in=state:{STATE_FIPS}"
    res = requests.get(url)
    res.raise_for_status()

    data = res.json()

    df = pd.DataFrame(data[1:], columns=data[0])

    df["county_fips"] = df["state"] + df["county"]

    counties = list(df["county_fips"])
    counties.sort()

    with open(COUNTIES_FILE, "w") as file:
        file.write("\n".join(counties))


def get_county_fips() -> list[str]:
    """
    Gets all county FIPS codes in the state for generating per-county tasks.
    This is memoized to avoid unnecessary API calls.
    """

    if not COUNTIES_FILE.exists():
        _download_county_fips()

    counties = []
    with open(COUNTIES_FILE, "r") as file:
        for line in file.readlines():
            counties.append(line.strip())

    return counties


def filter_df_by_county(df: pd.DataFrame, county_fips: str) -> pd.DataFrame:
    return df.loc[df["county_fips"] == county_fips]


def haversine(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray):
    """
    Computes haversize distance between numpy arrays of latitude and longitude
    coordinates.
    """
    R = 3956  # Radius of Earth in miles

    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))

    distance = R * c
    return distance


def census_api_call(api_vars: list[str]) -> pd.DataFrame:
    """
    Executes a Census API call. The Census API limits API calls to a maximum of
    50 API variables per call, so the `api_vars` inputted are split into chunks
    of 50, and the data for each call is merged together.
    """
    CHUNK_SIZE = 50
    dfs = []

    for i in range(0, len(api_vars), CHUNK_SIZE):
        chunk_api_vars = api_vars[i : i + CHUNK_SIZE]

        url = f"https://api.census.gov/data/{CENSUS_YEAR}/acs/acs5?get={','.join(chunk_api_vars)}&for=county:*&in=state:{STATE_FIPS}"

        res = requests.get(url)
        res.raise_for_status()

        data = res.json()
        df = pd.DataFrame(data[1:], columns=data[0])
        dfs.append(df)

    def merge_df(l_df, r_df):
        return pd.merge(l_df, r_df, on=["state", "county"])

    merged_df = reduce(merge_df, dfs)
    merged_df[api_vars] = merged_df[api_vars].astype("int32")
    return merged_df
