import random
from itertools import product
from typing import Annotated, Any, cast

import pandas as pd
import pytask
from pytask import task

from fred_pop_gen.config import DATA_CATALOG, RNG
from fred_pop_gen.constants import Enrollment, Grade
from fred_pop_gen.utils import get_county_fips, get_persons_in_household, haversine


for county in get_county_fips():

    @task(id=county)
    def task_assign_grade_to_persons_in_county(
        df: Annotated[pd.DataFrame, DATA_CATALOG[f"persons_{county}"]],
    ) -> Annotated[pd.DataFrame, DATA_CATALOG[f"persons_w_grade_{county}"]]:
        df["grade"] = df["agep"].apply(map_age_to_grade)
        return df.loc[~df["grade"].isnull()]

    @task(id=county)
    def get_households_for_school_assignment_in_county(
        hh_df: Annotated[pd.DataFrame, DATA_CATALOG[f"households_{county}"]],
        p_df: Annotated[pd.DataFrame, DATA_CATALOG[f"persons_w_grade_{county}"]],
    ) -> Annotated[
        pd.DataFrame, DATA_CATALOG[f"households_for_school_assignment_{county}"]
    ]:
        def check_household_for_children(household: pd.Series) -> bool:
            hh_persons = get_persons_in_household(household.name, p_df)

            # persons_w_grade only includes school aged children, so check that
            # there is at least one person in the household who is assigned
            # a grade
            return not hh_persons.index.intersection(p_df.index).empty

        hh_df = hh_df.loc[hh_df.apply(check_household_for_children, axis=1)]

        return hh_df

    @task(id=county)
    def assign_enrollment_to_households_in_county(
        county: Annotated[str, county],
        hh_df: Annotated[
            pd.DataFrame,
            DATA_CATALOG[f"households_for_school_assignment_{county}"],
        ],
        enrollment_df: Annotated[pd.DataFrame, DATA_CATALOG["enrollment_proportions"]],
    ) -> Annotated[pd.DataFrame, DATA_CATALOG[f"households_w_enrollment_{county}"]]:
        enrollment_probabilities = enrollment_df.loc[county]
        print(enrollment_probabilities)

        def assign_enrollment_to_household(household: pd.Series) -> Enrollment:
            choices = [Enrollment.PUBLIC, Enrollment.PRIVATE, Enrollment.NOT_ENROLLED]
            i = RNG.choice(len(choices), 1, p=enrollment_probabilities.values)[0]

            return choices[i]

        hh_df["enrollment"] = hh_df.apply(assign_enrollment_to_household, axis=1)

        return hh_df

    @task(id=county)
    def get_public_school_household_distances_in_county(
        hh_df: Annotated[
            pd.DataFrame,
            DATA_CATALOG[f"households_w_enrollment_{county}"],
        ],
        sch_df: Annotated[pd.DataFrame, DATA_CATALOG[f"public_schools_{county}"]],
    ) -> Annotated[pd.DataFrame, DATA_CATALOG[f"pubsch_hh_distance_{county}"]]:
        hh_df = hh_df.loc[hh_df["enrollment"] == Enrollment.PUBLIC]

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

    @task(id=county)
    def assign_public_schools_in_county(
        p_df: Annotated[pd.DataFrame, DATA_CATALOG[f"persons_w_grade_{county}"]],
        hh_df: Annotated[
            pd.DataFrame,
            DATA_CATALOG[f"households_w_enrollment_{county}"],
        ],
        pubsch_df: Annotated[pd.DataFrame, DATA_CATALOG[f"public_schools_{county}"]],
        dist_df: Annotated[pd.DataFrame, DATA_CATALOG[f"pubsch_hh_distance_{county}"]],
    ) -> Annotated[pd.DataFrame, DATA_CATALOG[f"persons_w_school_{county}"]]:
        # initialize household assignment statuses
        hh_df["assigned"] = False

        # initialize school assignment columns
        p_df["school_id"] = None

        # track current number of assigned students per school
        enrollment = {}

        # initialize enrollment to 0 for every school
        for sch_id in list(pubsch_df.index):
            enrollment[sch_id] = 0

        for edge in dist_df.itertuples():
            # cast to Any to avoid static typing issues with accessint
            # NamedTuple fields
            edge: Any = cast(Any, edge)

            pubsch = pubsch_df.loc[edge.sch_id]
            hh = hh_df.loc[edge.hh_id]

            # skip this edge if school is beyond enrollment capacity
            if enrollment[edge.sch_id] > pubsch["enrollment_total"]:
                continue

            # skip this edge if house has already been assigned
            if hh["assigned"]:
                continue

            # get the children in the household
            hh_persons = get_persons_in_household(edge.hh_id, p_df)

            # filter by children who have not yet been assigned
            hh_persons = hh_persons.loc[hh_persons["school_id"].isnull()]

            # filter by children who are eligible for the school
            def check_if_school_offers_students_grade(grade: Grade) -> bool:
                return Grade.is_grade_in_range(
                    grade, pubsch["lowest_grade"], pubsch["highest_grade"]
                )

            hh_persons_assigned = hh_persons[
                hh_persons["grade"].apply(check_if_school_offers_students_grade)
            ]

            # assign children to school
            p_df.loc[hh_persons_assigned.index, "school_id"] = edge.sch_id

            # increment enrollment counts
            enrollment[edge.sch_id] += len(hh_persons_assigned)

            # read in updated household children
            hh_persons = get_persons_in_household(edge.hh_id, p_df)

            # if all children in the household have been assigned, mark the
            # household as assigned
            if not hh_persons["school_id"].isnull().any():
                hh_df.loc[edge.hh_id, "assigned"] = True

        print(p_df)

        print("assigned:")
        print(p_df[~p_df["school_id"].isnull()])
        print("not assigned:")
        print(p_df[p_df["school_id"].isnull()])

        print(enrollment)
        print(pubsch_df["enrollment_total"])
        assert False

        return p_df


def map_age_to_grade(age: int) -> Grade | None:
    match age:
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
