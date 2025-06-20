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
    get_county_fips,
    haversine,
)


for _county in get_county_fips():

    @task(id=_county)
    def task_get_persons_for_public_school_assignment_in_county(
        county: Annotated[str, _county],
        p_df: Annotated[
            pd.DataFrame, DATA_CATALOG[f"persons_w_enrollment_{STATE_FIPS}"]
        ],
    ) -> Annotated[pd.DataFrame, DATA_CATALOG[f"persons_w_pub_enrollment_{_county}"]]:
        """
        Filters persons by county and public enrollment.
        """
        p_df = p_df.loc[p_df["county_fips"] == county]
        p_df = p_df.loc[p_df["enrollment"] == Enrollment.PUBLIC]

        return p_df

    @task(id=_county)
    def task_get_public_school_distances_in_county(
        p_df: Annotated[
            pd.DataFrame, DATA_CATALOG[f"persons_w_pub_enrollment_{_county}"]
        ],
        sch_df: Annotated[pd.DataFrame, DATA_CATALOG[f"public_schools_{_county}"]],
    ) -> Annotated[pd.DataFrame, DATA_CATALOG[f"public_hh_distance_{_county}"]]:
        """
        Gets the school distances for public schools by county.
        """
        return get_school_distances(p_df, sch_df)

    @task(id=_county)
    def task_assign_public_schools_in_county(
        p_df: Annotated[
            pd.DataFrame, DATA_CATALOG[f"persons_w_pub_enrollment_{_county}"]
        ],
        sch_df: Annotated[pd.DataFrame, DATA_CATALOG[f"public_schools_{_county}"]],
        dist_df: Annotated[pd.DataFrame, DATA_CATALOG[f"public_hh_distance_{_county}"]],
    ) -> Annotated[pd.DataFrame, DATA_CATALOG[f"persons_w_public_school_{_county}"]]:
        """
        Assigns public schools by county.
        """
        return assign_schools_to_persons(p_df, sch_df, dist_df)


def task_get_persons_for_private_school_assignment(
    p_df: Annotated[pd.DataFrame, DATA_CATALOG[f"persons_w_enrollment_{STATE_FIPS}"]],
) -> Annotated[pd.DataFrame, DATA_CATALOG[f"persons_w_priv_enrollment_{STATE_FIPS}"]]:
    """
    Filters persons by private enrollment.
    """
    p_df = p_df.loc[p_df["enrollment"] == Enrollment.PRIVATE]

    return p_df


def task_get_private_school_distances(
    p_df: Annotated[
        pd.DataFrame, DATA_CATALOG[f"persons_w_priv_enrollment_{STATE_FIPS}"]
    ],
    sch_df: Annotated[pd.DataFrame, DATA_CATALOG[f"private_schools_{STATE_FIPS}"]],
) -> Annotated[pd.DataFrame, DATA_CATALOG[f"private_school_distances_{STATE_FIPS}"]]:
    """
    Gets the school distances for private schools by state.
    """
    return get_school_distances(p_df, sch_df)


def task_assign_private_schools(
    p_df: Annotated[
        pd.DataFrame, DATA_CATALOG[f"persons_w_priv_enrollment_{STATE_FIPS}"]
    ],
    sch_df: Annotated[pd.DataFrame, DATA_CATALOG[f"private_schools_{STATE_FIPS}"]],
    dist_df: Annotated[
        pd.DataFrame, DATA_CATALOG[f"private_school_distances_{STATE_FIPS}"]
    ],
) -> Annotated[pd.DataFrame, DATA_CATALOG[f"persons_w_private_school_{STATE_FIPS}"]]:
    """
    Assigns private schools by state.
    """
    return assign_schools_to_persons(p_df, sch_df, dist_df)


def get_school_distances(p_df: pd.DataFrame, sch_df: pd.DataFrame) -> pd.DataFrame:
    """
    Finds the distance between every possible pair of person and school in
    the provided dataframes. The persons and school dataframes will form a
    complete bipartite graph. The resulting dataframe will have the id of
    the person and school forming the pair, as well as the distance between
    the person (which is merged from the household df into the person df in
    `task_merge_p_hh_df`) and the school.
    """
    p_idx, sch_idx = np.meshgrid(
        p_df.index.to_numpy(), sch_df.index.to_numpy(), indexing="ij"
    )
    p_idx = p_idx.ravel()
    sch_idx = sch_idx.ravel()

    p_lat = p_df.loc[p_idx, "lat"].to_numpy()
    p_lon = p_df.loc[p_idx, "lon"].to_numpy()
    sch_lat = sch_df.loc[sch_idx, "lat"].to_numpy()
    sch_lon = sch_df.loc[sch_idx, "lon"].to_numpy()

    distances = haversine(p_lat, p_lon, sch_lat, sch_lon)

    df = pd.DataFrame({"p_id": p_idx, "sch_id": sch_idx, "distance": distances})

    return df


def assign_schools_to_persons(
    p_df: pd.DataFrame, sch_df: pd.DataFrame, dist_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Assigns the provided persons to the provided schools using the distances
    computed in `get_school_distances`, which contain person-school pairs. The
    algorithm iterates through these pairs in order of increasing distance such
    that the persons closest to schools will be assigned first.

    Assignment to a school will occur if:
        - The school still has remaining capacity
        - The person has ot yet been assigned
        - The school offers the grade level of the person

    After the main loop, some post-processing is done to assign leftover
    students to schools. This is done to account for inconsistencies in the
    generated population, the computed enrollment proportions, and the reported
    school capacities.
    """
    p_df["school_id"] = None

    dist_df = dist_df.sort_values(by="distance", ascending=True)

    # track current number of assigned students per school
    enrollment = {sch_id: 0 for sch_id in sch_df.index.to_list()}

    # track assignments to apply at end of loop
    assignments = {}

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

        # check for early exit
        n_unassigned = len(p_df) - len(assignments)
        if n_unassigned == 0:
            break

    # compute a scale factor to ensure all students are assigned a school and
    # enrollment is evenly distributed
    capacity_scale_factor = len(p_df) / sch_df["enrollment_total"].sum()
    capacity_scale_factor += 0.1  # add some headroom
    if capacity_scale_factor < 1:
        capacity_scale_factor = 1

    # assign leftover students to nearest school
    def assign_person_to_nearest_school(p_id):
        for edge in dist_df.loc[dist_df["p_id"] == p_id].itertuples():
            # capacity = (
            #     sch_df.loc[edge.sch_id, "enrollment_total"] * capacity_scale_factor
            # )
            # if enrollment[edge.sch_id] > capacity:
            #     continue

            if p_df.loc[p_id, "grade"] not in eligible_grades[edge.sch_id]:
                continue

            assignments.update({p_id: edge.sch_id})
            enrollment[edge.sch_id] += 1
            break

    unassigned_df = p_df[~p_df.index.isin(assignments)]
    unassigned_df.index.map(assign_person_to_nearest_school)

    p_df["school_id"] = p_df.index.map(assignments.get)

    # TODO: handle case where there are no schools that offer PREK in county,
    # for now, we will leave them unassigned, as the numbers aren't too large
    # and it is unlikely there would be PREK kids attending out of county
    # schools
    #
    # TODO: handle case where there are a large amount of private school
    # enrolees are leftover, in some states (such as WY), there is much less
    # private school capacity available then private school enrollees

    # unassigned_df = p_df[p_df["school_id"].isnull()]

    # some helpful debug statements:
    #
    # print(capacity_scale_factor)
    # sch_df["enrollment"] = sch_df.index.map(lambda x: enrollment[x])
    # print(sch_df[["lowest_grade", "highest_grade", "enrollment", "enrollment_total"]])
    # print(unassigned_df[["grade"]])
    # assert unassigned_df.empty

    return p_df
