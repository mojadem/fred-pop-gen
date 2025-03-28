from pathlib import Path
from typing import Annotated

import pandas as pd
import pytask
import requests
from pytask import Product

from fred_pop_gen.config import CENSUS_YEAR, DATA, DATA_CATALOG, STATE_FIPS


# below are the ACS variables needed from the census API
# reference: https://api.census.gov/data/2019/acs/acs5/variables.html

# these columns represent school-aged children enrolled in public schools
PUBLIC_SCHOOL_COLS = [
    "B14003_004E",  # male ages 3-4
    "B14003_005E",  # male ages 5-9
    "B14003_006E",  # male ages 10-14
    "B14003_007E",  # male ages 15-17
    "B14003_032E",  # female ages 3-4
    "B14003_033E",  # female ages 5-9
    "B14003_034E",  # female ages 10-14
    "B14003_035E",  # female ages 15-17
]

# these columns represent school-aged children enrolled in private schools
PRIVATE_SCHOOL_COLS = [
    "B14003_013E",  # male ages 3-4
    "B14003_014E",  # male ages 5-9
    "B14003_015E",  # male ages 10-14
    "B14003_016E",  # male ages 15-17
    "B14003_041E",  # female ages 3-4
    "B14003_042E",  # female ages 5-9
    "B14003_043E",  # female ages 10-14
    "B14003_044E",  # female ages 15-17
]

# these columns represent school-aged children not enrolled in either public
# or private schools
NOT_ENROLLED_COLS = [
    "B14003_022E",  # male ages 3-4
    "B14003_023E",  # male ages 5-9
    "B14003_024E",  # male ages 10-14
    "B14003_025E",  # male ages 15-17
    "B14003_050E",  # female ages 3-4
    "B14003_051E",  # female ages 5-9
    "B14003_052E",  # female ages 10-14
    "B14003_053E",  # female ages 15-17
]

API_VARS = PUBLIC_SCHOOL_COLS + PRIVATE_SCHOOL_COLS + NOT_ENROLLED_COLS


@pytask.mark.persist
def task_get_enrollment_census_data(
    path: Annotated[Path, Product] = DATA / f"input/enrollment-data-{STATE_FIPS}.pkl",
) -> None:
    """Saves enrollment data from the Census API."""

    df = pd.DataFrame()

    url = f"https://api.census.gov/data/{CENSUS_YEAR}/acs/acs5?get={','.join(API_VARS)}&for=county:*&in=state:{STATE_FIPS}"
    res = requests.get(url)
    res.raise_for_status()

    data = res.json()
    df = pd.DataFrame(data[1:], columns=data[0])
    df[API_VARS] = df[API_VARS].astype("int32")

    df.to_pickle(path)


def task_generate_enrollment_totals(
    path: Path = DATA / f"input/enrollment-data-{STATE_FIPS}.pkl",
) -> Annotated[pd.DataFrame, DATA_CATALOG[f"enrollment_totals_{STATE_FIPS}"]]:
    """Generates totals for each enrollment status."""

    totals_df = pd.read_pickle(path)
    df = pd.DataFrame()

    df["county_fips"] = totals_df["state"] + totals_df["county"]
    df["public_total"] = totals_df[PUBLIC_SCHOOL_COLS].sum(axis=1)
    df["private_total"] = totals_df[PRIVATE_SCHOOL_COLS].sum(axis=1)
    df["not_enrolled_total"] = totals_df[NOT_ENROLLED_COLS].sum(axis=1)

    df = df.set_index("county_fips")
    df["total"] = df.sum(axis=1)

    return df


def task_generate_enrollment_proportions(
    totals_df: Annotated[pd.DataFrame, DATA_CATALOG[f"enrollment_totals_{STATE_FIPS}"]],
) -> Annotated[pd.DataFrame, DATA_CATALOG[f"enrollment_proportions_{STATE_FIPS}"]]:
    """Generates proportions of each enrollment status using enrollment totals."""

    df = pd.DataFrame()

    df.index = totals_df.index

    df["public"] = totals_df["public_total"] / totals_df["total"]
    df["private"] = totals_df["private_total"] / totals_df["total"]
    df["not_enrolled"] = totals_df["not_enrolled_total"] / totals_df["total"]

    return df
