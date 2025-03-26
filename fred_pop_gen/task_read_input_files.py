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
)
from fred_pop_gen.constants import Grade


def task_read_persons_file(
    path: Path = PERSONS_FILE,
) -> Annotated[pd.DataFrame, DATA_CATALOG["persons"]]:
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
) -> Annotated[pd.DataFrame, DATA_CATALOG["households"]]:
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
) -> Annotated[pd.DataFrame, DATA_CATALOG["public_schools"]]:
    """Loads the public schools file into a DataFrame."""

    df = pd.read_csv(path)
    df = strip_school_file_column_names(df)

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
) -> Annotated[pd.DataFrame, DATA_CATALOG["private_schools"]]:
    """Loads the private schools file into a DataFrame."""

    df = pd.read_csv(path)
    df = strip_school_file_column_names(df)

    cols = df.columns.tolist()
    expected_cols = [
        "Private School Name",
        "State Name",
        "School ID - NCES Assigned",
        "ANSI/FIPS County Code",
        "Lowest Grade Taught",
        "Highest Grade Taught",
        "Total Students (Ungraded & PK-12)",
    ]
    assert sorted(cols) == sorted(expected_cols), (
        f"private schools file did not contain expected columns: expected = {sorted(expected_cols)}, actual = {sorted(cols)}"
    )

    column_map = {
        "School ID - NCES Assigned": "id",
        "ANSI/FIPS County Code": "county_fips",
        "Lowest Grade Taught": "lowest_grade",
        "Highest Grade Taught": "highest_grade",
        "Total Students (Ungraded & PK-12)": "enrollment_total",
    }
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


def strip_school_file_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    The public and private school files contain a suffix ('[Public School]...'
    or '[Private School]...') on all column names. Here, we simply strip that suffix
    out as a preprocessing step.
    """

    for column in df.columns:
        column = str(column)
        stripped = re.sub(r" \[.*$", "", column)
        df = df.rename(columns={column: stripped})

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


def map_grade_level(grade: str) -> Grade | None:
    grade = grade.lower().strip()

    match grade:
        case "prekindergarten":
            return Grade.PREK
        case "kindergarten":
            return Grade.K
        case "transitional kindergarten":
            return Grade.K
        case "1st grade":
            return Grade.FIRST
        case "2nd grade":
            return Grade.SECOND
        case "3rd grade":
            return Grade.THIRD
        case "4th grade":
            return Grade.FOURTH
        case "5th grade":
            return Grade.FIFTH
        case "6th grade":
            return Grade.SIXTH
        case "7th grade":
            return Grade.SEVENTH
        case "8th grade":
            return Grade.EIGHTH
        case "9th grade":
            return Grade.NINTH
        case "10th grade":
            return Grade.TENTH
        case "11th grade":
            return Grade.ELEVENTH
        case "12th grade":
            return Grade.TWELFTH
        case _:
            return None
