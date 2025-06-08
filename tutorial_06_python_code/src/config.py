# make sure these states stay constant across different code implementations
from pathlib import Path 

# Path object to determine to project root folder
PROJECT_ROOT = Path(__file__).resolve().parents[1]
# data path
DATA_PATH = PROJECT_ROOT / "data" / "housing.csv"
# data splitting random state
DATA_RANDOM_STATE = 0
# model random state
MODEL_RANDOM_STATE = 0
# test/train ratio
TEST_SIZE = 0.2