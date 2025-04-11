from pathlib import Path

from pytask import DataCatalog
import numpy as np

DATA = Path(__file__).parent.joinpath("..", "data").resolve()
DATA_CATALOG = DataCatalog()
RNG = np.random.default_rng(0)

# TODO: paramaterize and support multiple of each file
STATE_FIPS = "10"
STATE_ABBR = "DE"
CENSUS_YEAR = 2019

MAX_DISTANCE_FOR_PUBLIC_SCHOOL_HOUSEHOLD_PAIRS_KM = 50
SCHOOL_ENROLLMENT_CAPACITY_FACTOR = 1.2

PERSONS_FILE = DATA / f"input/{STATE_FIPS}_{CENSUS_YEAR}_persons.parquet"
HOUSEHOLDS_FILE = DATA / f"input/{STATE_ABBR}_{CENSUS_YEAR}_households_w_geom.parquet"
PUBLIC_SCHOOLS_FILE = DATA / "input/public-schools.csv"
PRIVATE_SCHOOLS_FILE = DATA / "input/private-schools.csv"
