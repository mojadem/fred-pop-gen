from typing import Annotated, Dict

from fred_pop_gen.config import DATA_CATALOG, RNG, STATE_FIPS
from fred_pop_gen.constants import EmploymentAgeBucket
from fred_pop_gen.utils import get_county_fips
import pandas as pd
from pytask import task


for _county in get_county_fips():

    @task(id=_county)
    def task_assign_employment_to_persons_in_county(
        county: Annotated[str, _county],
        p_df: Annotated[pd.DataFrame, DATA_CATALOG[f"persons_{_county}"]],
        employment: Annotated[
            Dict[str, pd.DataFrame],
            DATA_CATALOG[f"employment_proportions_{STATE_FIPS}"],
        ],
    ) -> Annotated[pd.DataFrame, DATA_CATALOG[f"persons_w_employment_{_county}"]]:
        print(p_df)

        def generate_random_employment(person: pd.Series) -> bool:
            age = int(person["agep"])
            sex = "male" if person["sex"] == 1 else "female"

            bucket = EmploymentAgeBucket.get_bucket(age)

            if bucket == EmploymentAgeBucket.B_UNDER_16:
                return False

            p = employment[sex].loc[county][bucket]
            return RNG.random() < p

        p_df["employed"] = p_df.apply(generate_random_employment, axis=1)
        return p_df
