from typing import Annotated, Any, cast

import pandas as pd
import numpy as np
from pytask import task

from fred_pop_gen.config import (
    DATA_CATALOG,
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
    def task_get_public_school_household_distances_in_county(
        p_df: Annotated[
            pd.DataFrame, DATA_CATALOG[f"persons_w_enrollment_{STATE_FIPS}"]
        ],
        hh_df: Annotated[
            pd.DataFrame,
            DATA_CATALOG[f"households_{_county}"],
        ],
        sch_df: Annotated[pd.DataFrame, DATA_CATALOG[f"public_schools_{_county}"]],
    ) -> Annotated[pd.DataFrame, DATA_CATALOG[f"public_hh_distance_{_county}"]]:
        p_df = p_df.loc[p_df["enrollment"] == Enrollment.PUBLIC]
        p_hh_df = merge_p_hh_df(p_df, hh_df)

        return get_school_distances(p_hh_df, sch_df)

    @task(id=_county)
    def task_assign_public_schools_in_county(
        p_df: Annotated[
            pd.DataFrame, DATA_CATALOG[f"persons_w_enrollment_{STATE_FIPS}"]
        ],
        sch_df: Annotated[pd.DataFrame, DATA_CATALOG[f"public_schools_{_county}"]],
        dist_df: Annotated[pd.DataFrame, DATA_CATALOG[f"public_hh_distance_{_county}"]],
    ) -> Annotated[pd.DataFrame, DATA_CATALOG[f"persons_w_public_school_{_county}"]]:
        p_df = p_df.loc[p_df["enrollment"] == Enrollment.PUBLIC]

        return assign_schools_to_persons(p_df, sch_df, dist_df)


def task_get_private_school_distances(
    p_df: Annotated[pd.DataFrame, DATA_CATALOG[f"persons_w_enrollment_{STATE_FIPS}"]],
    hh_df: Annotated[
        pd.DataFrame,
        DATA_CATALOG[f"households_{STATE_FIPS}"],
    ],
    sch_df: Annotated[pd.DataFrame, DATA_CATALOG[f"private_schools_{STATE_FIPS}"]],
) -> Annotated[pd.DataFrame, DATA_CATALOG[f"private_school_distances_{STATE_FIPS}"]]:
    p_df = p_df.loc[p_df["enrollment"] == Enrollment.PRIVATE]
    p_hh_df = merge_p_hh_df(p_df, hh_df)

    return get_school_distances(p_hh_df, sch_df)


def task_assign_private_schools(
    p_df: Annotated[pd.DataFrame, DATA_CATALOG[f"persons_w_enrollment_{STATE_FIPS}"]],
    sch_df: Annotated[pd.DataFrame, DATA_CATALOG[f"private_schools_{STATE_FIPS}"]],
    dist_df: Annotated[
        pd.DataFrame, DATA_CATALOG[f"private_school_distances_{STATE_FIPS}"]
    ],
) -> Annotated[pd.DataFrame, DATA_CATALOG[f"persons_w_private_school_{STATE_FIPS}"]]:
    p_df = p_df.loc[p_df["enrollment"] == Enrollment.PRIVATE]

    return assign_schools_to_persons(p_df, sch_df, dist_df)


def merge_p_hh_df(p_df: pd.DataFrame, hh_df: pd.DataFrame) -> pd.DataFrame:
    p_hh_df = p_df.merge(hh_df, on="hh_id", how="left")

    # we must retain the orignal person index as they function as each person's
    # unique id
    assert len(p_df) == len(p_hh_df)
    p_hh_df = p_hh_df.set_index(p_df.index)

    return p_hh_df


def get_school_distances(p_hh_df: pd.DataFrame, sch_df: pd.DataFrame) -> pd.DataFrame:
    p_idx, sch_idx = np.meshgrid(
        p_hh_df.index.to_numpy(), sch_df.index.to_numpy(), indexing="ij"
    )
    p_idx = p_idx.ravel()
    sch_idx = sch_idx.ravel()

    p_lat = p_hh_df.loc[p_idx, "lat"].to_numpy()
    p_lon = p_hh_df.loc[p_idx, "lon"].to_numpy()
    sch_lat = sch_df.loc[sch_idx, "lat"].to_numpy()
    sch_lon = sch_df.loc[sch_idx, "lon"].to_numpy()

    distances = haversine(p_lat, p_lon, sch_lat, sch_lon)

    df = pd.DataFrame({"p_id": p_idx, "sch_id": sch_idx, "distance": distances})
    df = df.sort_values(by="distance", ascending=True)

    return df


def assign_schools_to_persons(
    p_df: pd.DataFrame, sch_df: pd.DataFrame, dist_df: pd.DataFrame
) -> pd.DataFrame:
    p_df["school_id"] = None

    # track current number of assigned students per school
    enrollment = {sch_id: 0 for sch_id in sch_df.index.to_list()}

    # track assignments to apply at end of loop
    assignments = {}

    n = 0

    # precompile eligible grades per school
    eligible_grades = {
        sch_id: Grade.range(Grade(sch["lowest_grade"]), Grade(sch["highest_grade"]))
        for sch_id, sch in sch_df.iterrows()
    }

    for edge in dist_df.itertuples():
        # cast to Any to avoid static typing issues with accessing
        # NamedTuple fields
        edge: Any = cast(Any, edge)

        # skip if school is at capacity
        capacity = sch_df.loc[edge.sch_id, "enrollment_total"]
        if enrollment[edge.sch_id] > capacity:
            continue

        # skip if person was already assigned
        if edge.p_id in assignments:
            continue

        # skip if person is not eligible for school
        if p_df.loc[edge.p_id, "grade"] not in eligible_grades[edge.sch_id]:
            continue

        # assign
        assignments.update({edge.p_id: edge.sch_id})
        enrollment[edge.sch_id] += 1
        n += 1

        # check for early exit
        n_unassigned = len(p_df) - len(assignments)
        if n_unassigned == 0:
            break

    # compute a scale factor to ensure all students are assigned a school and
    # enrollment is evenly distributed
    capacity_scale_factor = len(p_df) / sch_df["enrollment_total"].sum()
    capacity_scale_factor += 0.1  # add some headroom

    # assign leftover students to nearest school
    def assign_person_to_nearest_school(p_id):
        for edge in dist_df.loc[dist_df["p_id"] == p_id].itertuples():
            capacity = (
                sch_df.loc[edge.sch_id, "enrollment_total"] * capacity_scale_factor
            )
            if enrollment[edge.sch_id] > capacity:
                continue

            if p_df.loc[p_id, "grade"] not in eligible_grades[edge.sch_id]:
                continue

            assignments.update({p_id: edge.sch_id})
            enrollment[edge.sch_id] += 1
            break

    unassigned_df = p_df[~p_df.index.isin(assignments)]
    unassigned_df.index.map(assign_person_to_nearest_school)

    p_df["school_id"] = p_df.index.map(assignments.get)

    # TODO: handle case where there are no schools that offer PREK in county
    unassigned_df = p_df[p_df["school_id"].isnull()]
    assert unassigned_df.empty

    return p_df
