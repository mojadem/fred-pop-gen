from typing import Annotated

import pandas as pd

from fred_pop_gen.config import (
    DATA_CATALOG,
    RNG,
    STATE_FIPS,
)
from fred_pop_gen.constants import Enrollment, Grade


def task_assign_grade_to_persons(
    p_df: Annotated[pd.DataFrame, DATA_CATALOG[f"persons_w_geo_{STATE_FIPS}"]],
) -> Annotated[pd.DataFrame, DATA_CATALOG[f"persons_w_grade_{STATE_FIPS}"]]:
    """
    Maps persons' age to grade level, filtering out non-school-aged persons.
    """
    p_df["grade"] = p_df["agep"].apply(map_age_to_grade)

    # drop non-school-aged people
    p_df = p_df.loc[p_df["grade"].notna()]

    return p_df


def task_assign_enrollment_to_persons(
    p_df: Annotated[pd.DataFrame, DATA_CATALOG[f"persons_w_grade_{STATE_FIPS}"]],
    hh_df: Annotated[pd.DataFrame, DATA_CATALOG[f"households_{STATE_FIPS}"]],
    enrollment_df: Annotated[
        pd.DataFrame, DATA_CATALOG[f"enrollment_proportions_{STATE_FIPS}"]
    ],
) -> Annotated[pd.DataFrame, DATA_CATALOG[f"persons_w_enrollment_{STATE_FIPS}"]]:
    def generate_random_enrollment(person: pd.Series) -> Enrollment:
        hh_id = str(person["hh_id"])
        assert hh_id in hh_df.index

        county = str(person["county_fips"])
        enrollment_probabilities = enrollment_df.loc[county]

        p_non_prek = [
            enrollment_probabilities["public"],
            enrollment_probabilities["private"],
            enrollment_probabilities["not_enrolled"],
        ]
        p_prek = [
            enrollment_probabilities["public_prek"],
            enrollment_probabilities["private_prek"],
            enrollment_probabilities["not_enrolled_prek"],
        ]

        choices = [Enrollment.PUBLIC, Enrollment.PRIVATE, Enrollment.NOT_ENROLLED]

        grade = person["grade"]

        p = p_prek if grade == Grade.PREK else p_non_prek
        i = RNG.choice(len(choices), 1, p=p)[0]
        return choices[i]

    p_df["enrollment"] = p_df.apply(generate_random_enrollment, axis=1)
    return p_df


def map_age_to_grade(age: int) -> Grade | None:
    match age:
        case 3:
            return Grade.PREK
        case 4:
            return Grade.PREK
        case 5:
            return Grade.K
        case 6:
            return Grade.FIRST
        case 7:
            return Grade.SECOND
        case 8:
            return Grade.THIRD
        case 9:
            return Grade.FOURTH
        case 10:
            return Grade.FIFTH
        case 11:
            return Grade.SIXTH
        case 12:
            return Grade.SEVENTH
        case 13:
            return Grade.EIGHTH
        case 14:
            return Grade.NINTH
        case 15:
            return Grade.TENTH
        case 16:
            return Grade.ELEVENTH
        case 17:
            return Grade.TWELFTH
        case _:
            return None
