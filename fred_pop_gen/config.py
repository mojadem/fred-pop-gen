from pathlib import Path

from pytask import DataCatalog

DATA = Path(__file__).parent.joinpath("..", "data").resolve()
DATA_CATALOG = DataCatalog()

# TODO: paramaterize and support multiple of each file
STATE_FIPS = 56
STATE_ABBR = "WY"
CENSUS_YEAR = 2019

PERSONS_FILE = DATA / f"input/{STATE_FIPS}_{CENSUS_YEAR}_persons.csv"
HOUSEHOLDS_FILE = DATA / f"input/{STATE_ABBR}_{CENSUS_YEAR}_households_w_geom.csv"
