import re
from pathlib import Path
from typing import Annotated

import pandas as pd

from fred_pop_gen.config import (
    DATA_CATALOG,
    HOUSEHOLDS_FILE,
    PERSONS_FILE,
    PRIVATE_SCHOOLS_FILE,
    PUBLIC_SCHOOLS_FILE,
    STATE_FIPS,
)
from fred_pop_gen.constants import Grade


def task_read_persons_file(
    path: Path = PERSONS_FILE,
) -> Annotated[pd.DataFrame, DATA_CATALOG[f"persons_{STATE_FIPS}"]]:
    """Loads the persons file into a DataFrame."""

    df = pd.read_parquet(path)

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
) -> Annotated[pd.DataFrame, DATA_CATALOG[f"households_{STATE_FIPS}"]]:
    """Loads the households file into a DataFrame."""

    df = pd.read_parquet(path)

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
    df = format_df(df, column_map)

    return df


def task_read_public_schools_file(
    path: Path = PUBLIC_SCHOOLS_FILE,
) -> Annotated[pd.DataFrame, DATA_CATALOG[f"public_schools_{STATE_FIPS}"]]:
    """Loads the public schools file into a DataFrame."""

    df = pd.read_csv(path)

    # remove suffix ([Public School]...) from column names
    rename_columns = {}
    for column in df.columns:
        column = str(column)
        stripped = re.sub(r" \[.*$", "", column)
        rename_columns.update({column: stripped})
    df = df.rename(columns=rename_columns)

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
    df = format_df(df, column_map, drop=True)
    df = post_format_schools_df(df)

    return df


def task_read_private_schools_file(
    path: Path = PRIVATE_SCHOOLS_FILE,
) -> Annotated[pd.DataFrame, DATA_CATALOG[f"private_schools_{STATE_FIPS}"]]:
    """Loads the private schools file into a DataFrame."""

    df = pd.read_csv(path)

    fips_cols = [
        "PSTANSI",
        "PCNTY",
    ]
    df[fips_cols] = df[fips_cols].astype(str)
    df["county_fips"] = df["PSTANSI"].str.zfill(2) + df["PCNTY"].str.zfill(3)

    column_map = {
        "PPIN": "id",
        "county_fips": "county_fips",
        "LONGITUDE": "lon",
        "LATITUDE": "lat",
        "LOGR": "lowest_grade",
        "HIGR": "highest_grade",
        "P305": "enrollment_total",
    }

    # remove suffix from column names
    rename_columns = {
        original: stripped
        for original in df.columns
        for stripped in column_map
        if original.startswith(stripped)
    }
    df = df.rename(columns=rename_columns)

    for col in column_map:
        assert col in df.columns, (
            f"public schools file did not contain expected column: expected = {col}"
        )

    df = format_df(df, column_map, drop=True)
    df = post_format_schools_df(df)

    return df


def format_df(df: pd.DataFrame, column_map: dict[str, str], drop=False) -> pd.DataFrame:
    """
    Formats a DataFrame by renaming columns based on a provided mapping and
    optionally dropping columns that are not in the mapping.
    """

    if drop:
        df = df.drop(columns=[col for col in df.columns if col not in column_map])

    df = df.rename(columns=column_map)

    return df


def post_format_schools_df(df: pd.DataFrame) -> pd.DataFrame:
    df["lowest_grade"] = df["lowest_grade"].apply(map_grade_level)
    df["highest_grade"] = df["highest_grade"].apply(map_grade_level)

    df["county_fips"] = df["county_fips"].astype("string")
    df["enrollment_total"] = pd.to_numeric(df["enrollment_total"], errors="coerce")
    df = df.set_index("id")

    # TODO: should we recover schools with bad valuees instead of just dropping?
    df = df.dropna()

    return df


def map_grade_level(grade: str | int) -> Grade | None:
    """
    Maps the grade level in school files to a Grade enum. The public school file
    uses string values while the private school file uses int values.
    """
    if type(grade) is str:
        grade = grade.lower().strip()

    match grade:
        case "prekindergarten" | 2:
            return Grade.PREK
        case "kindergarten" | 3:
            return Grade.K
        case "transitional kindergarten" | 4:
            return Grade.K
        case "1st grade" | 5 | 6:  # 5 is transitional first grade
            return Grade.FIRST
        case "2nd grade" | 7:
            return Grade.SECOND
        case "3rd grade" | 8:
            return Grade.THIRD
        case "4th grade" | 9:
            return Grade.FOURTH
        case "5th grade" | 10:
            return Grade.FIFTH
        case "6th grade" | 11:
            return Grade.SIXTH
        case "7th grade" | 12:
            return Grade.SEVENTH
        case "8th grade" | 13:
            return Grade.EIGHTH
        case "9th grade" | 14:
            return Grade.NINTH
        case "10th grade" | 15:
            return Grade.TENTH
        case "11th grade" | 16:
            return Grade.ELEVENTH
        case "12th grade" | 17:
            return Grade.TWELFTH
        case _:
            return None
