import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / 'data' / 'movies.json'

def load_movies() -> list[dict]:
    try:
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        return data["movies"]

    except json.JSONDecodeError as e:
        print(f"JSON Error: {e}")
        print(f"Line: {e.lineno}")
        print(f"Column: {e.colno}")
        print(f"Position: {e.pos}")
        raise