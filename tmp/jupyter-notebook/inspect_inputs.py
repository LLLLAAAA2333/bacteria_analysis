from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"


parquet_path = DATA / "106bac.parquet"
parquet_file = pq.ParquetFile(parquet_path)
print("PARQUET METADATA")
print("rows:", parquet_file.metadata.num_rows)
print("row_groups:", parquet_file.metadata.num_row_groups)
print("schema:")
print(parquet_file.schema)

sample = pd.read_parquet(parquet_path).head(8)
print("\nPARQUET SAMPLE")
print(sample.to_string(index=False))
print("\nPARQUET DTYPES")
print(sample.dtypes.to_string())

for path in sorted(DATA.glob("*.xlsx")):
    print(f"\nWORKBOOK: {path.name}")
    book = pd.ExcelFile(path)
    print("sheets:", book.sheet_names)
    for sheet in book.sheet_names:
        frame = pd.read_excel(path, sheet_name=sheet)
        print(f"  SHEET {sheet!r}: shape={frame.shape}")
        print("  columns:", [repr(column) for column in frame.columns])
        print(frame.head(3).to_string(index=False))

