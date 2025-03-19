from typing import Annotated

import pandas as pd
from pytask import task

from fred_pop_gen.config import DATA_CATALOG
from fred_pop_gen.utils import filter_df_by_county, get_county_fips

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
