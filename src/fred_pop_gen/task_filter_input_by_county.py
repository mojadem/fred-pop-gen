from typing import Annotated

import pandas as pd
from pytask import task

from fred_pop_gen.config import DATA_CATALOG, STATE_FIPS
from fred_pop_gen.utils import filter_df_by_county, get_county_fips

for _county in get_county_fips():

    @task(id=_county)
    def get_households_in_county(
        county: Annotated[str, _county],
        df: Annotated[pd.DataFrame, DATA_CATALOG[f"households_{STATE_FIPS}"]],
    ) -> Annotated[pd.DataFrame, DATA_CATALOG[f"households_{_county}"]]:
        return filter_df_by_county(df, county)

    @task(id=_county)
    def get_persons_in_county(
        p_df: Annotated[pd.DataFrame, DATA_CATALOG[f"persons_{STATE_FIPS}"]],
        hh_df: Annotated[pd.DataFrame, DATA_CATALOG[f"households_{_county}"]],
    ) -> Annotated[pd.DataFrame, DATA_CATALOG[f"persons_{_county}"]]:
        return p_df.loc[p_df["hh_id"].apply(lambda x: x in hh_df.index)]

    @task(id=_county)
    def get_public_schools_in_county(
        county: Annotated[str, _county],
        df: Annotated[pd.DataFrame, DATA_CATALOG[f"public_schools_{STATE_FIPS}"]],
    ) -> Annotated[pd.DataFrame, DATA_CATALOG[f"public_schools_{_county}"]]:
        return filter_df_by_county(df, county)

    @task(id=_county)
    def get_private_schools_in_county(
        county: Annotated[str, _county],
        df: Annotated[pd.DataFrame, DATA_CATALOG[f"private_schools_{STATE_FIPS}"]],
    ) -> Annotated[pd.DataFrame, DATA_CATALOG[f"private_schools_{_county}"]]:
        return filter_df_by_county(df, county)
