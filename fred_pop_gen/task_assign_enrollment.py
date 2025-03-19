import random
from pathlib import Path
from typing import Annotated

import pandas as pd
import pytask
import requests
from pytask import Product

from fred_pop_gen.config import CENSUS_YEAR, DATA, DATA_CATALOG, STATE_FIPS
from fred_pop_gen.constants import Enrollment

"""
In this task, we assign enrollment to school-aged persons. Enrollment is assigned
by marking a person as either enrolled in a public school, enrolled in a private
school, or not enrolled in either public or private.

We do this by generating proportions of the count of school-aged persons with
each enrollment status to the total count of school-aged persons. This is done
per county.

School-aged in this case is ages 3-18. Note that the census data has ages 18-19 as an age bucket; this is divided by 2 for our counts.
"""

# below are the ACS variables needed from the census API
# reference: https://api.census.gov/data/2019/acs/acs5/variables.html

# these columns represent school-aged children enrolled in public schools
PUBLIC_SCHOOL_COLS = [
    "B14003_004E",  # male ages 3-4
    "B14003_005E",  # male ages 5-9
    "B14003_006E",  # male ages 10-14
    "B14003_007E",  # male ages 15-17
    "B14003_008E",  # male ages 18-19
    "B14003_032E",  # female ages 3-4
    "B14003_033E",  # female ages 5-9
    "B14003_034E",  # female ages 10-14
    "B14003_035E",  # female ages 15-17
    "B14003_036E",  # female ages 18-19
]

# these columns represent school-aged children enrolled in private schools
PRIVATE_SCHOOL_COLS = [
    "B14003_013E",  # male ages 3-4
    "B14003_014E",  # male ages 5-9
    "B14003_015E",  # male ages 10-14
    "B14003_016E",  # male ages 15-17
    "B14003_017E",  # male ages 18-19
    "B14003_041E",  # female ages 3-4
    "B14003_042E",  # female ages 5-9
    "B14003_043E",  # female ages 10-14
    "B14003_044E",  # female ages 15-17
    "B14003_045E",  # female ages 18-19
]

# these columns represent school-aged children not enrolled in either public
# or private schools
NOT_ENROLLED_COLS = [
    "B14003_022E",  # male ages 3-4
    "B14003_023E",  # male ages 5-9
    "B14003_024E",  # male ages 10-14
    "B14003_025E",  # male ages 15-17
    "B14003_026E",  # male ages 18-19
    "B14003_050E",  # female ages 3-4
    "B14003_051E",  # female ages 5-9
    "B14003_052E",  # female ages 10-14
    "B14003_053E",  # female ages 15-17
    "B14003_054E",  # female ages 18-19
]

API_VARS = PUBLIC_SCHOOL_COLS + PRIVATE_SCHOOL_COLS + NOT_ENROLLED_COLS


@pytask.mark.persist
def task_get_enrollment_census_data(
    path: Annotated[Path, Product] = DATA / "input/enrollment-data.pkl",
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
    path: Path = DATA / "input/enrollment-data.pkl",
) -> Annotated[pd.DataFrame, DATA_CATALOG["enrollment_totals"]]:
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
    totals_df: Annotated[pd.DataFrame, DATA_CATALOG["enrollment_totals"]],
) -> Annotated[pd.DataFrame, DATA_CATALOG["enrollment_proportions"]]:
    """Generates proportions of each enrollment status using enrollment totals."""

    df = pd.DataFrame()

    df.index = totals_df.index

    df["public"] = totals_df["public_total"] / totals_df["total"]
    df["private"] = totals_df["private_total"] / totals_df["total"]
    df["not_enrolled"] = totals_df["not_enrolled_total"] / totals_df["total"]

    return df


def task_assign_enrollment(
    persons_df: Annotated[pd.DataFrame, DATA_CATALOG["persons"]],
    households_df: Annotated[pd.DataFrame, DATA_CATALOG["households"]],
    enrollment_df: Annotated[pd.DataFrame, DATA_CATALOG["enrollment_proportions"]],
) -> Annotated[pd.DataFrame, DATA_CATALOG["persons_w_enrollment"]]:
    """
    Assigns enrollment to persons using enrollment probabilities.

    Enrollment is sampled per-household, so all school-aged children in a
    household will share an enrollment status.
    """

    def assign_enrollment_to_household(household: pd.Series) -> Enrollment:
        county = household["county_fips"]
        enrollment_probabilities = enrollment_df.loc[county]

        choices = [Enrollment.PUBLIC, Enrollment.PRIVATE, Enrollment.NOT_ENROLLED]
        enrollment = random.choices(choices, enrollment_probabilities)[0]

        return enrollment

    households_df["enrollment"] = households_df.apply(
        assign_enrollment_to_household, axis=1
    )

    def assign_enrollment_to_person(person: pd.Series) -> Enrollment:
        if person["agep"] < 3 or person["agep"] > 18:
            return Enrollment.NOT_SCHOOL_AGED

        hh_id = person["hh_id"]
        household = households_df.loc[hh_id]
        return household["enrollment"]

    persons_df["enrollment"] = persons_df.apply(assign_enrollment_to_person, axis=1)

    return persons_df
