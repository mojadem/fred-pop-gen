from typing import Annotated
from fred_pop_gen.config import DATA_CATALOG, STATE_FIPS
import pandas as pd


def task_merge_p_hh_df(
    p_df: Annotated[pd.DataFrame, DATA_CATALOG[f"persons_{STATE_FIPS}"]],
    hh_df: Annotated[pd.DataFrame, DATA_CATALOG[f"households_{STATE_FIPS}"]],
) -> Annotated[pd.DataFrame, DATA_CATALOG[f"persons_w_geo_{STATE_FIPS}"]]:
    """
    Merges the persons df with the households df such that each person row also
    contains its corresponding household's columns. This is useful to associate
    persons with their household's latitude, longitude, and county.
    """
    p_hh_df = p_df.merge(hh_df, on="hh_id", how="left")

    # we must retain the orignal person index as they function as each person's
    # unique id
    assert len(p_df) == len(p_hh_df)
    p_hh_df = p_hh_df.set_index(p_df.index)

    return p_hh_df
