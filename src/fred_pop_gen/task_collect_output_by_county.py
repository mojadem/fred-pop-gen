from pathlib import Path
from typing import Annotated

import pandas as pd
from pytask import DirectoryNode, Product, task

from fred_pop_gen.config import DATA, DATA_CATALOG, STATE_FIPS
from fred_pop_gen.utils import get_county_fips


SCHOOL_ROOT_DIR = DATA / "interim" / f"school_{STATE_FIPS}"

for _county in get_county_fips():

    @task(id=_county)
    def task_serialize_school_output_in_county(
        county: Annotated[str, _county],
        df: Annotated[pd.DataFrame, DATA_CATALOG[f"persons_w_public_school_{_county}"]],
        dir: Annotated[
            Path,
            DirectoryNode(root_dir=SCHOOL_ROOT_DIR, pattern=_county),
            Product,
        ],
    ) -> None:
        df.to_pickle(dir.joinpath(f"{county}.pkl"))


@task(after="serialize_school_output_in_county")
def task_collect_school_output(
    pubsch_paths: Annotated[
        list[Path], DirectoryNode(root_dir=SCHOOL_ROOT_DIR, pattern="*")
    ],
    privsch_df: Annotated[
        pd.DataFrame, DATA_CATALOG[f"persons_w_private_school_{STATE_FIPS}"]
    ],
    p_df: Annotated[pd.DataFrame, DATA_CATALOG[f"persons_{STATE_FIPS}"]],
) -> Annotated[pd.DataFrame, DATA_CATALOG[f"persons_w_school_{STATE_FIPS}"]]:
    pubsch_df = pd.concat([pd.read_pickle(path) for path in pubsch_paths])
    sch_df = pd.concat([pubsch_df, privsch_df])

    p_df["school_id"] = sch_df["school_id"]

    return p_df
