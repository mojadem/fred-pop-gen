from itertools import product
from typing import Annotated

import pandas as pd
import requests
from pytask import task

from fred_pop_gen.config import CENSUS_YEAR, DATA_CATALOG, STATE_FIPS
from fred_pop_gen.constants import Enrollment
from fred_pop_gen.utils import filter_df_by_county, haversine


def get_county_fips() -> list[str]:
    url = f"https://api.census.gov/data/{CENSUS_YEAR}/acs/acs5?get=B01001_001E&for=county:*&in=state:{STATE_FIPS}"
    res = requests.get(url)
    res.raise_for_status()

    data = res.json()

    df = pd.DataFrame(data[1:], columns=data[0])

    df["county_fips"] = df["state"] + df["county"]

    return list(df["county_fips"])


# TODO: add function comments
for county in get_county_fips():

    @task(id=county)
    def get_households_in_county(
        county: Annotated[str, county],
        df: Annotated[pd.DataFrame, DATA_CATALOG["households"]],
    ) -> Annotated[pd.DataFrame, DATA_CATALOG[f"households_{county}"]]:
        return filter_df_by_county(df, county)

    @task(id=county)
    def get_persons_in_county(
        p_df: Annotated[pd.DataFrame, DATA_CATALOG["persons"]],
        hh_df: Annotated[pd.DataFrame, DATA_CATALOG[f"households_{county}"]],
    ) -> Annotated[pd.DataFrame, DATA_CATALOG[f"persons_{county}"]]:
        return p_df.loc[p_df["hh_id"].apply(lambda x: x in hh_df.index)]

    @task(id=county)
    def get_public_schools_in_county(
        county: Annotated[str, county],
        df: Annotated[pd.DataFrame, DATA_CATALOG["public_schools"]],
    ) -> Annotated[pd.DataFrame, DATA_CATALOG[f"public_schools_{county}"]]:
        return filter_df_by_county(df, county)

    @task(id=county)
    def get_private_schools_in_county(
        county: Annotated[str, county],
        df: Annotated[pd.DataFrame, DATA_CATALOG["private_schools"]],
    ) -> Annotated[pd.DataFrame, DATA_CATALOG[f"private_schools_{county}"]]:
        return filter_df_by_county(df, county)

    @task(id=county)
    def get_public_school_household_distances(
        hh_df: Annotated[pd.DataFrame, DATA_CATALOG[f"households_{county}"]],
        sch_df: Annotated[pd.DataFrame, DATA_CATALOG[f"public_schools_{county}"]],
    ) -> Annotated[pd.DataFrame, DATA_CATALOG[f"pubsch_hh_distance_{county}"]]:
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

        return distance_df

    @task(id=county)
    def assign_public_schools_in_county(
        p_df: Annotated[pd.DataFrame, DATA_CATALOG[f"persons_{county}"]],
        hh_df: Annotated[pd.DataFrame, DATA_CATALOG[f"households_{county}"]],
        pubsch_df: Annotated[pd.DataFrame, DATA_CATALOG[f"public_schools_{county}"]],
    ) -> None:
        print("people in county:", len(p_df))
        print("households in county:", len(hh_df))
        print("schools in county:", len(pubsch_df))
        assert False


# def task_test(
#     persons_df: Annotated[pd.DataFrame, DATA_CATALOG["persons_w_enrollment"]],
#     pub_schools_df: Annotated[pd.DataFrame, DATA_CATALOG["public_schools"]],
#     priv_schools_df: Annotated[pd.DataFrame, DATA_CATALOG["private_schools"]],
# ) -> None:
#     print(
#         "public enrollees",
#         len(persons_df[persons_df["enrollment"] == Enrollment.PUBLIC]),
#     )
#     print("public enrollment totals", pub_schools_df["enrollment_total"].sum())

#     print(
#         "private enrollees",
#         len(persons_df[persons_df["enrollment"] == Enrollment.PRIVATE]),
#     )
#     print("private enrollment totals", priv_schools_df["enrollment_total"].sum())


#     assert False
#     pass
#
