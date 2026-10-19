from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent  # script in src/
DATA_DIR = ROOT / "data"
OUT_FILE = DATA_DIR / "bts_2022_2026_combined.parquet"

files = sorted(DATA_DIR.glob("bts_20??_combined.parquet"))
print(f"{len(files)} files found in {DATA_DIR}")
if not files:
    raise SystemExit("No yearly files found. Check the path.")

schema = pq.ParquetFile(files[0]).schema_arrow  # first file defines the schema
total = 0

with pq.ParquetWriter(OUT_FILE, schema, compression="snappy") as writer:
    for f in files:
        pf = pq.ParquetFile(f)
        for batch in pf.iter_batches(batch_size=500_000):
            table = pa.Table.from_batches([batch])
            if table.schema != schema:
                table = table.select(schema.names).cast(
                    schema
                )  # align column order/types
            writer.write_table(table)
            total += table.num_rows
        print(f"ok {f.name}")

print(f"{total:,} rows -> {OUT_FILE}")
