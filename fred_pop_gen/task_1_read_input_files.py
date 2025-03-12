from pathlib import Path
from typing import Annotated

import pandas as pd

from fred_pop_gen.config import HOUSEHOLDS_FILE, PERSONS_FILE, DATA_CATALOG


def task_read_persons_file(
    path: Path = PERSONS_FILE,
) -> Annotated[pd.DataFrame, DATA_CATALOG["persons_df"]]:
    """Loads the persons file into a DataFrame."""
    df = pd.read_csv(path, index_col=0)

    cols = df.columns.tolist()
    expected_cols = [
        "hh_id",
        "serialno",
        "sporder",
        "rac1p",
        "agep",
        "sex",
        "relshipp",
    ]
    assert cols == expected_cols, (
        f"persons file did not contain expected columns: expected = {expected_cols}, actual = {cols}"
    )

    return df


def task_read_households_file(
    path: Path = HOUSEHOLDS_FILE,
) -> Annotated[pd.DataFrame, DATA_CATALOG["households_df"]]:
    """Loads the households file into a DataFrame."""
    df = pd.read_csv(path, index_col="hh_id")

    cols = df.columns.tolist()
    expected_cols = [
        "GEOID",
        "geometry",
        "lon_4326",
        "lat_4326",
        "hh_age",
        "hh_income",
        "hh_race",
        "size",
        "serialno",
        "state_fips",
        "puma_fips",
        "county_fips",
        "tract_fips",
        "blkgrp_fips",
    ]
    assert cols == expected_cols, (
        f"households file did not contain expected columns: expected = {expected_cols}, actual = {cols}"
    )

    return df
