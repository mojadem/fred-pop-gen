from pathlib import Path
from typing import Annotated

import pandas as pd
import pytask
from pytask import Product
import requests

from fred_pop_gen.config import CENSUS_YEAR, DATA, DATA_CATALOG, STATE_ABBR, STATE_FIPS

LODES_BASE_URL = f"https://lehd.ces.census.gov/data/lodes/LODES8/{STATE_ABBR.lower()}"

OD_FILE_NAME = f"{STATE_ABBR.lower()}_od_main_JT00_{CENSUS_YEAR}.csv.gz"
OD_FILE_PATH = DATA / f"input/{OD_FILE_NAME}"
OD_FILE_URL = f"{LODES_BASE_URL}/od/{OD_FILE_NAME}"

WAC_FILE_NAME = f"{STATE_ABBR.lower()}_wac_S000_JT00_{CENSUS_YEAR}.csv.gz"
WAC_FILE_PATH = DATA / f"input/{WAC_FILE_NAME}"
WAC_FILE_URL = f"{LODES_BASE_URL}/wac/{WAC_FILE_NAME}"


@pytask.mark.persist
def task_download_lodes_od_file(
    path: Annotated[Path, Product] = OD_FILE_PATH,
) -> None:
    """
    Downloads the LODES OD (Origin-Destination) file to disk.
    """
    download_file(OD_FILE_URL, path)


@pytask.mark.persist
def task_download_lodes_wac_file(
    path: Annotated[Path, Product] = WAC_FILE_PATH,
) -> None:
    """
    Downloads the LODES WAC (Workplace Area Characteristics) file to disk.
    """
    download_file(WAC_FILE_URL, path)


def task_read_lodes_od_file(
    path: Path = OD_FILE_PATH,
) -> Annotated[pd.DataFrame, DATA_CATALOG[f"lodes_od_{STATE_FIPS}"]]:
    df = pd.read_csv(path, compression="gzip")

    return df


def task_read_lodes_wac_file(
    path: Path = WAC_FILE_PATH,
) -> Annotated[pd.DataFrame, DATA_CATALOG[f"lodes_wac_{STATE_FIPS}"]]:
    df = pd.read_csv(path, compression="gzip")

    return df


def download_file(url: str, path: Path) -> None:
    res = requests.get(url)

    with open(path, "wb") as file:
        file.write(res.content)
