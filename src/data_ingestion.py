from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).parent
INGESTED_DIR = BASE_DIR / "ingested_"
INPUT_FILE = BASE_DIR.parent / "data.csv"
OUTPUT_FILE = INGESTED_DIR / "data.csv"

def ingest_data():
    INGESTED_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"{INPUT_FILE} is not found.")

    df = pd.read_csv(INPUT_FILE)
    assert not df.empty, "DataFrame is empty."

    df.to_csv(OUTPUT_FILE, index=False)
    print(f'Data ingested successfully and saved to {OUTPUT_FILE}')

    return df

if __name__ == "__main__":
    ingest_data()