from pathlib import Path

from pytask import DataCatalog
import numpy as np

# TODO: paramaterize and support multiple of each file
STATE_FIPS = "56"
STATE_ABBR = "WY"
CENSUS_YEAR = 2019

SEED = 123

SRC = Path(__file__).parent.resolve()
DATA = SRC.joinpath("..", "..", "data").resolve()

DATA_CATALOG = DataCatalog()
RNG = np.random.default_rng(SEED)

PERSONS_FILE = DATA / f"input/{STATE_FIPS}_{CENSUS_YEAR}_persons.parquet"
HOUSEHOLDS_FILE = DATA / f"input/{STATE_ABBR}_{CENSUS_YEAR}_households_w_geom.parquet"
PUBLIC_SCHOOLS_FILE = DATA / "input/public-schools.csv"
PRIVATE_SCHOOLS_FILE = DATA / "input/private-schools.csv"
