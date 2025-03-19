import numpy as np
import pandas as pd


def filter_df_by_county(df: pd.DataFrame, county_fips: str) -> pd.DataFrame:
    return df.loc[df["county_fips"] == county_fips]


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
