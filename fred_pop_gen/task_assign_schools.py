from typing import Annotated, Any, cast

import pandas as pd
import numpy as np
from pytask import task

from fred_pop_gen.config import (
    DATA_CATALOG,
    MAX_DISTANCE_FOR_PUBLIC_SCHOOL_HOUSEHOLD_PAIRS_KM,
    RNG,
    SCHOOL_ENROLLMENT_CAPACITY_FACTOR,
    STATE_FIPS,
)
from fred_pop_gen.constants import Enrollment, Grade
from fred_pop_gen.utils import (
    filter_households_by_resident_enrollment,
    get_county_fips,
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

        hh_idx, sch_idx = np.meshgrid(
            hh_df.index.to_numpy(), sch_df.index.to_numpy(), indexing="ij"
        )
        hh_idx = hh_idx.ravel()
        sch_idx = sch_idx.ravel()

        hh_lat: np.ndarray = hh_df.loc[hh_idx, "lat"].to_numpy()
        hh_lon: np.ndarray = hh_df.loc[hh_idx, "lon"].to_numpy()
        sch_lat: np.ndarray = sch_df.loc[sch_idx, "lat"].to_numpy()
        sch_lon: np.ndarray = sch_df.loc[sch_idx, "lon"].to_numpy()

        distances = haversine(hh_lat, hh_lon, sch_lat, sch_lon)

        df = pd.DataFrame(
            {"hh_id": hh_idx.ravel(), "sch_id": sch_idx.ravel(), "distance": distances}
        )

        df = df.loc[df["distance"] < MAX_DISTANCE_FOR_PUBLIC_SCHOOL_HOUSEHOLD_PAIRS_KM]
        df = df.sort_values(by="distance", ascending=True)

        return df

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

        hh_person_groups = p_df.groupby("hh_id")

        # track current number of assigned students per school
        enrollment = {}

        # track assignments to apply at end of loop
        assignments = {}

        # precompile eligible grades per school
        eligible_grades = {
            sch_id: set(
                Grade.range(Grade(sch["lowest_grade"]), Grade(sch["highest_grade"]))
            )
            for sch_id, sch in pubsch_df.iterrows()
        }

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

            # get persons in current household
            persons_to_assign = pd.DataFrame(
                hh_person_groups.get_group(edge.hh_id)
            ).index

            # filter by children who have not yet been assigned
            persons_to_assign = persons_to_assign[
                persons_to_assign.map(lambda x: x not in assignments)
            ]

            # filter by children who are eligible for the school
            persons_to_assign = persons_to_assign[
                persons_to_assign.map(
                    lambda x: p_df.loc[x, "grade"] in eligible_grades[edge.sch_id]
                )
            ]

            # assign
            assignments.update({p_id: edge.sch_id for p_id in persons_to_assign})
            enrollment[edge.sch_id] += len(persons_to_assign)

            # check for early exit
            n_unassigned = len(p_df) - len(assignments)
            if n_unassigned == 0:
                break

        p_df["school_id"] = p_df.index.map(assignments.get)

        unassigned_df = p_df[p_df["school_id"].isna()]
        # print(len(p_df) - len(assignments))
        # print(unassigned_df["grade"])
        # print(enrollment)
        # print(pubsch_df[["lowest_grade", "highest_grade", "enrollment_total"]])
        # unassigned_df = unassigned_df.loc[unassigned_df["grade"] != Grade.PREK]
        print(len(unassigned_df))
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
