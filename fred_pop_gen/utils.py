from typing import Hashable
import numpy as np
import pandas as pd
import requests

from fred_pop_gen.config import CENSUS_YEAR, STATE_FIPS, DATA
from fred_pop_gen.constants import Enrollment

COUNTIES_FILE = DATA / f"input/counties-{STATE_FIPS}.txt"


def _download_county_fips() -> None:
    """
    Downloads county FIPS codes from the census API and writes them to
    COUNTIES_FILE.
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


def get_persons_in_household(hh_id: Hashable, p_df: pd.DataFrame) -> pd.DataFrame:
    return p_df.loc[p_df["hh_id"] == hh_id]


def filter_persons_by_household_enrollment(
    p_df: pd.DataFrame, hh_df: pd.DataFrame, enrollment: Enrollment
) -> pd.DataFrame:
    hh_df = hh_df.loc[hh_df["enrollment"] == enrollment]
    return p_df.loc[p_df["hh_id"].isin(list(hh_df.index))]


def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the distance between two points on Earth using the Haversine formula.

    Parameters:
    lat1, lon1: Latitude and longitude of the first point in degrees.
    lat2, lon2: Latitude and longitude of the second point in degrees.

    Returns:
    Distance in kilometers.
    """

    R = 3956  # Radius of Earth in miles

    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))

    distance = R * c
    return distance
