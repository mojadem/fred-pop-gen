from pathlib import Path

from pytask import DataCatalog

DATA = Path(__file__).parent.joinpath("..", "data").resolve()

# TODO: paramaterize and support multiple of each file
PERSONS_FILE = DATA / "input/56_2019_persons.csv"
HOUSEHOLDS_FILE = DATA / "input/WY_2019_households_w_geom.csv"

DATA_CATALOG = DataCatalog()
