from pathlib import Path
from typing import Annotated, Dict

from fred_pop_gen.constants import EmploymentAgeBucket
import pandas as pd
import pytask
from pytask import Product

from fred_pop_gen.config import DATA, DATA_CATALOG, STATE_FIPS
from fred_pop_gen.utils import census_api_call

MALE_TOTAL_COLS = [
    "B23001_003E",  # male ages 16-19 - total
    "B23001_010E",  # male ages 20-21 - total
    "B23001_017E",  # male ages 22-24 - total
    "B23001_024E",  # male ages 25-29 - total
    "B23001_031E",  # male ages 30-34 - total
    "B23001_038E",  # male ages 35-44 - total
    "B23001_045E",  # male ages 45-54 - total
    "B23001_052E",  # male ages 55-59 - total
    "B23001_059E",  # male ages 60-61 - total
    "B23001_066E",  # male ages 62-64 - total
    "B23001_073E",  # male ages 65-69 - total
    "B23001_078E",  # male ages 70-74 - total
    "B23001_083E",  # male ages 75+ - total
]

MALE_EMPLOYED_COLS = [
    "B23001_007E",  # male ages 16-19 - employed
    "B23001_014E",  # male ages 20-21 - employed
    "B23001_021E",  # male ages 22-24 - employed
    "B23001_028E",  # male ages 25-29 - employed
    "B23001_035E",  # male ages 30-34 - employed
    "B23001_042E",  # male ages 35-44 - employed
    "B23001_049E",  # male ages 45-54 - employed
    "B23001_056E",  # male ages 55-59 - employed
    "B23001_063E",  # male ages 60-61 - employed
    "B23001_070E",  # male ages 62-64 - employed
    "B23001_075E",  # male ages 65-69 - employed
    "B23001_080E",  # male ages 70-74 - employed
    "B23001_085E",  # male ages 75+ - employed
]


FEMALE_TOTAL_COLS = [
    "B23001_089E",  # female ages 16-19 - total
    "B23001_096E",  # female ages 20-21 - total
    "B23001_103E",  # female ages 22-24 - total
    "B23001_110E",  # female ages 25-29 - total
    "B23001_117E",  # female ages 30-34 - total
    "B23001_124E",  # female ages 35-44 - total
    "B23001_131E",  # female ages 45-54 - total
    "B23001_138E",  # female ages 55-59 - total
    "B23001_145E",  # female ages 60-61 - total
    "B23001_152E",  # female ages 62-64 - total
    "B23001_159E",  # female ages 65-69 - total
    "B23001_164E",  # female ages 70-74 - total
    "B23001_169E",  # female ages 75+ - total
]

FEMALE_EMPLOYED_COLS = [
    "B23001_093E",  # female ages 16-19 - employed
    "B23001_100E",  # female ages 20-21 - employed
    "B23001_107E",  # female ages 22-24 - employed
    "B23001_114E",  # female ages 25-29 - employed
    "B23001_121E",  # female ages 30-34 - employed
    "B23001_128E",  # female ages 35-44 - employed
    "B23001_135E",  # female ages 45-54 - employed
    "B23001_142E",  # female ages 55-59 - employed
    "B23001_149E",  # female ages 60-61 - employed
    "B23001_156E",  # female ages 62-64 - employed
    "B23001_161E",  # female ages 65-69 - employed
    "B23001_166E",  # female ages 70-74 - employed
    "B23001_171E",  # female ages 75+ - employed
]

CENSUS_VAR_BUCKET_MAP = {
    "B23001_003E": EmploymentAgeBucket.B_16_TO_19,
    "B23001_007E": EmploymentAgeBucket.B_16_TO_19,
    "B23001_010E": EmploymentAgeBucket.B_20_TO_21,
    "B23001_014E": EmploymentAgeBucket.B_20_TO_21,
    "B23001_017E": EmploymentAgeBucket.B_22_TO_24,
    "B23001_021E": EmploymentAgeBucket.B_22_TO_24,
    "B23001_024E": EmploymentAgeBucket.B_25_TO_29,
    "B23001_028E": EmploymentAgeBucket.B_25_TO_29,
    "B23001_031E": EmploymentAgeBucket.B_30_TO_34,
    "B23001_035E": EmploymentAgeBucket.B_30_TO_34,
    "B23001_038E": EmploymentAgeBucket.B_35_TO_44,
    "B23001_042E": EmploymentAgeBucket.B_35_TO_44,
    "B23001_045E": EmploymentAgeBucket.B_45_TO_54,
    "B23001_049E": EmploymentAgeBucket.B_45_TO_54,
    "B23001_052E": EmploymentAgeBucket.B_55_TO_59,
    "B23001_056E": EmploymentAgeBucket.B_55_TO_59,
    "B23001_059E": EmploymentAgeBucket.B_60_TO_61,
    "B23001_063E": EmploymentAgeBucket.B_60_TO_61,
    "B23001_066E": EmploymentAgeBucket.B_62_TO_64,
    "B23001_070E": EmploymentAgeBucket.B_62_TO_64,
    "B23001_073E": EmploymentAgeBucket.B_65_TO_69,
    "B23001_075E": EmploymentAgeBucket.B_65_TO_69,
    "B23001_078E": EmploymentAgeBucket.B_70_TO_74,
    "B23001_080E": EmploymentAgeBucket.B_70_TO_74,
    "B23001_083E": EmploymentAgeBucket.B_75_PLUS,
    "B23001_085E": EmploymentAgeBucket.B_75_PLUS,
    "B23001_089E": EmploymentAgeBucket.B_16_TO_19,
    "B23001_093E": EmploymentAgeBucket.B_16_TO_19,
    "B23001_096E": EmploymentAgeBucket.B_20_TO_21,
    "B23001_100E": EmploymentAgeBucket.B_20_TO_21,
    "B23001_103E": EmploymentAgeBucket.B_22_TO_24,
    "B23001_107E": EmploymentAgeBucket.B_22_TO_24,
    "B23001_110E": EmploymentAgeBucket.B_25_TO_29,
    "B23001_114E": EmploymentAgeBucket.B_25_TO_29,
    "B23001_117E": EmploymentAgeBucket.B_30_TO_34,
    "B23001_121E": EmploymentAgeBucket.B_30_TO_34,
    "B23001_124E": EmploymentAgeBucket.B_35_TO_44,
    "B23001_128E": EmploymentAgeBucket.B_35_TO_44,
    "B23001_131E": EmploymentAgeBucket.B_45_TO_54,
    "B23001_135E": EmploymentAgeBucket.B_45_TO_54,
    "B23001_138E": EmploymentAgeBucket.B_55_TO_59,
    "B23001_142E": EmploymentAgeBucket.B_55_TO_59,
    "B23001_145E": EmploymentAgeBucket.B_60_TO_61,
    "B23001_149E": EmploymentAgeBucket.B_60_TO_61,
    "B23001_152E": EmploymentAgeBucket.B_62_TO_64,
    "B23001_156E": EmploymentAgeBucket.B_62_TO_64,
    "B23001_159E": EmploymentAgeBucket.B_65_TO_69,
    "B23001_161E": EmploymentAgeBucket.B_65_TO_69,
    "B23001_164E": EmploymentAgeBucket.B_70_TO_74,
    "B23001_166E": EmploymentAgeBucket.B_70_TO_74,
    "B23001_169E": EmploymentAgeBucket.B_75_PLUS,
    "B23001_171E": EmploymentAgeBucket.B_75_PLUS,
}

API_VARS = (
    MALE_TOTAL_COLS + MALE_EMPLOYED_COLS + FEMALE_TOTAL_COLS + FEMALE_EMPLOYED_COLS
)


@pytask.mark.persist
def task_get_employment_census_data(
    path: Annotated[Path, Product] = DATA / f"input/employment-data-{STATE_FIPS}.pkl",
) -> None:
    df = census_api_call(API_VARS)

    df.to_pickle(path)


@pytask.mark.wip
def task_generate_employment_proportions(
    path: Path = DATA / f"input/employment-data-{STATE_FIPS}.pkl",
) -> Annotated[
    Dict[str, pd.DataFrame], DATA_CATALOG[f"employment_proportions_{STATE_FIPS}"]
]:
    totals_df = pd.read_pickle(path)
    assert isinstance(totals_df, pd.DataFrame)

    male_df = generate_proportions(MALE_EMPLOYED_COLS, MALE_TOTAL_COLS, totals_df)
    female_df = generate_proportions(FEMALE_EMPLOYED_COLS, FEMALE_TOTAL_COLS, totals_df)

    return {"male": male_df, "female": female_df}


def generate_proportions(
    employed_cols: list[str], total_cols: list[str], totals_df: pd.DataFrame
) -> pd.DataFrame:
    df = pd.DataFrame()
    df["county_fips"] = totals_df["state"] + totals_df["county"]

    for e_col, t_col in zip(employed_cols, total_cols):
        assert CENSUS_VAR_BUCKET_MAP[e_col] == CENSUS_VAR_BUCKET_MAP[t_col]
        bucket = CENSUS_VAR_BUCKET_MAP[e_col]
        df[bucket] = totals_df[e_col] / totals_df[t_col]

    df = df.set_index("county_fips")

    return df
