import re
from pathlib import Path
from typing import Annotated

import pandas as pd

from fred_pop_gen.config import (
    DATA_CATALOG,
    HOUSEHOLDS_FILE,
    PERSONS_FILE,
    PUBLIC_SCHOOLS_FILE,
)
from fred_pop_gen.constants import Grade


def task_read_persons_file(
    path: Path = PERSONS_FILE,
) -> Annotated[pd.DataFrame, DATA_CATALOG["persons"]]:
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
) -> Annotated[pd.DataFrame, DATA_CATALOG["households"]]:
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

    fips_cols = [
        "state_fips",
        "puma_fips",
        "county_fips",
        "tract_fips",
        "blkgrp_fips",
    ]
    df[fips_cols] = df[fips_cols].astype("string")

    column_map = {
        "lat_4326": "lat",
        "lon_4326": "lon",
    }
    df = df.rename(columns=column_map)

    return df


def task_read_public_schools_file(
    path: Path = PUBLIC_SCHOOLS_FILE,
) -> Annotated[pd.DataFrame, DATA_CATALOG["public_schools"]]:
    """Loads the public schools file into a DataFrame."""

    df = pd.read_csv(path)

    # preprocess column names to remove '[Public School]...'
    for column in df.columns:
        column = str(column)
        stripped = re.sub(r" \[Public School\].*$", "", column)
        df = df.rename(columns={column: stripped})

    cols = df.columns.tolist()
    expected_cols = [
        "School Name",
        "State Name",
        "School ID (12-digit) - NCES Assigned",
        "County Number",
        "Latitude",
        "Longitude",
        "Lowest Grade Offered",
        "Highest Grade Offered",
        "Total Students All Grades (Excludes AE)",
    ]
    assert sorted(cols) == sorted(expected_cols), (
        f"public schools file did not contain expected columns: expected = {sorted(expected_cols)}, actual = {sorted(cols)}"
    )

    column_map = {
        "School ID (12-digit) - NCES Assigned": "id",
        "County Number": "county_fips",
        "Latitude": "lat",
        "Longitude": "lon",
        "Lowest Grade Offered": "lowest_grade",
        "Highest Grade Offered": "highest_grade",
        "Total Students All Grades (Excludes AE)": "enrollment_total",
    }
    df = df.rename(columns=column_map)
    df = df.drop(columns=["School Name", "State Name"])

    df["lowest_grade"] = df["lowest_grade"].apply(map_grade_level)
    df["highest_grade"] = df["highest_grade"].apply(map_grade_level)

    return df


def map_grade_level(grade: str) -> Grade:
    match grade:
        case "Prekindergarten":
            return Grade.PREK
        case "Kindergarten":
            return Grade.K
        case "1st Grade":
            return Grade.FIRST
        case "2nd Grade":
            return Grade.SECOND
        case "3rd Grade":
            return Grade.THIRD
        case "4th Grade":
            return Grade.FOURTH
        case "5th Grade":
            return Grade.FIFTH
        case "6th Grade":
            return Grade.SIXTH
        case "7th Grade":
            return Grade.SEVENTH
        case "8th Grade":
            return Grade.EIGHTH
        case "9th Grade":
            return Grade.NINTH
        case "10th Grade":
            return Grade.TENTH
        case "11th Grade":
            return Grade.ELEVENTH
        case "12th Grade":
            return Grade.TWELFTH
        case _:
            raise ValueError(f"Unknown grade: {grade}")
