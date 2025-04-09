from itertools import product
from typing import Annotated, Any, cast

import pandas as pd
from pytask import task

from fred_pop_gen.config import (
    DATA_CATALOG,
    RNG,
    SCHOOL_ENROLLMENT_CAPACITY_FACTOR,
    STATE_FIPS,
)
from fred_pop_gen.constants import Enrollment, Grade
from fred_pop_gen.utils import (
    filter_households_by_resident_enrollment,
    get_county_fips,
    get_persons_in_household,
    haversine,
)

for _county in get_county_fips():

    @task(id=_county)
    def task_assign_grade_and_enrollment_to_persons_in_county(
        county: Annotated[str, _county],
        p_df: Annotated[pd.DataFrame, DATA_CATALOG[f"persons_{_county}"]],
        enrollment_df: Annotated[
            pd.DataFrame, DATA_CATALOG[f"enrollment_proportions_{STATE_FIPS}"]
        ],
    ) -> Annotated[pd.DataFrame, DATA_CATALOG[f"persons_w_grade_enrollment_{_county}"]]:
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

        p_df["grade"] = p_df["agep"].apply(map_age_to_grade)

        # drop non-school-aged people
        p_df = p_df.loc[p_df["grade"].notna()]

        def generate_random_enrollment(grade: Grade) -> Enrollment:
            choices = [Enrollment.PUBLIC, Enrollment.PRIVATE, Enrollment.NOT_ENROLLED]

            p = p_prek if grade == Grade.PREK else p_non_prek
            i = RNG.choice(len(choices), 1, p=p)[0]

            return choices[i]

        p_df["enrollment"] = p_df["grade"].apply(generate_random_enrollment)

        return p_df

    @task(id=_county)
    def get_public_school_household_distances_in_county(
        p_df: Annotated[
            pd.DataFrame, DATA_CATALOG[f"persons_w_grade_enrollment_{_county}"]
        ],
        hh_df: Annotated[
            pd.DataFrame,
            DATA_CATALOG[f"households_{_county}"],
        ],
        sch_df: Annotated[pd.DataFrame, DATA_CATALOG[f"public_schools_{_county}"]],
    ) -> Annotated[pd.DataFrame, DATA_CATALOG[f"pubsch_hh_distance_{_county}"]]:
        hh_df = filter_households_by_resident_enrollment(p_df, hh_df, Enrollment.PUBLIC)

        hh_ids = hh_df.index.values
        sch_ids = sch_df.index.values

        hh_sch_pairs = list(product(hh_ids, sch_ids))
        cols = pd.Index(["hh_id", "sch_id"])
        distance_df = pd.DataFrame(hh_sch_pairs, columns=cols)

        def distance(hh_id: int, sch_id: int) -> float:
            hh = hh_df.loc[hh_id]
            sch = sch_df.loc[sch_id]
            return haversine(hh["lat"], hh["lon"], sch["lat"], sch["lon"])

        distance_df["distance"] = distance_df.apply(
            lambda x: distance(x["hh_id"], x["sch_id"]), axis=1
        )

        return distance_df.sort_values(by="distance", ascending=True)

    @task(id=_county)
    def assign_public_schools_in_county(
        p_df: Annotated[
            pd.DataFrame, DATA_CATALOG[f"persons_w_grade_enrollment_{_county}"]
        ],
        pubsch_df: Annotated[pd.DataFrame, DATA_CATALOG[f"public_schools_{_county}"]],
        dist_df: Annotated[pd.DataFrame, DATA_CATALOG[f"pubsch_hh_distance_{_county}"]],
    ) -> Annotated[pd.DataFrame, DATA_CATALOG[f"persons_w_school_{_county}"]]:
        p_df = p_df.loc[p_df["enrollment"] == Enrollment.PUBLIC]
        p_df["school_id"] = pd.NA

        # track current number of assigned students per school
        enrollment = {}

        for sch_id in list(pubsch_df.index):
            enrollment[sch_id] = 0

        for edge in dist_df.itertuples():
            # cast to Any to avoid static typing issues with accessing
            # NamedTuple fields
            edge: Any = cast(Any, edge)

            sch = pubsch_df.loc[edge.sch_id]

            capacity = sch["enrollment_total"] * SCHOOL_ENROLLMENT_CAPACITY_FACTOR
            if enrollment[edge.sch_id] > capacity:
                continue

            hh_persons = get_persons_in_household(edge.hh_id, p_df)

            # filter by children who have not yet been assigned
            hh_persons = hh_persons.loc[hh_persons["school_id"].isnull()]

            # filter by children who are eligible for the school
            def check_if_school_offers_students_grade(grade: Grade) -> bool:
                return Grade.is_grade_in_range(
                    grade, sch["lowest_grade"], sch["highest_grade"]
                )

            hh_persons = hh_persons[
                hh_persons["grade"].apply(check_if_school_offers_students_grade)
            ]

            p_df.loc[hh_persons.index, "school_id"] = edge.sch_id
            enrollment[edge.sch_id] += len(hh_persons)

        unassigned_df = p_df[p_df["school_id"].isna()]
        assert unassigned_df.empty

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
