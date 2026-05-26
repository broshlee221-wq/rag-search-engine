import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT/'data'
MOVIES_PATH = DATA_PATH /'movies.json'
STOPWORD_PATH = DATA_PATH /'stopwords.txt'

CACHE_PATH = PROJECT_ROOT/'cache'


def load_movies() -> list[dict]:
    try:
        with open(MOVIES_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        return data["movies"]

    except json.JSONDecodeError as e:
        print(f"JSON Error: {e}")
        print(f"Line: {e.lineno}")
        print(f"Column: {e.colno}")
        print(f"Position: {e.pos}")
        raise

def load_stopwords() : 
    with open(STOPWORD_PATH, "r", encoding="utf-8") as f:
        data = f.read().splitlines()
    return data