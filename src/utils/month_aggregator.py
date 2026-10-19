from pathlib import Path
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent  # el script está en src/
DATA = ROOT / "data"
YEARS = range(2022, 2027)

COLS = [
    # Objetivo
    "DepDel15",
    # Predictoras
    "FlightDate",
    "Year",
    "Reporting_Airline",  # Added it in
    "Quarter",
    "Month",
    "DayofMonth",
    "DayOfWeek",
    "Origin",
    "Dest",
    "OriginState",
    "DestState",
    "CRSDepTime",
    "DepTimeBlk",
    "CRSArrTime",
    "CRSElapsedTime",
    "Distance",
    "Flight_Number_Reporting_Airline",
    "Tail_Number",
    # Control
    "Cancelled",
    "CancellationCode",
    "Diverted",
    # No predictoras (patrones históricos y filtros)
    "DepTime",
    "DepDelay",
    "DepDelayMinutes",
    "DepartureDelayGroups",
    "TaxiOut",
    "WheelsOff",
    "ArrDelay",
    "ArrDel15",
    "CarrierDelay",
    "WeatherDelay",
    "NASDelay",
    "SecurityDelay",
    "LateAircraftDelay",
]


def read_zip(zip_path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as z:
        csv_name = next(n for n in z.namelist() if n.lower().endswith(".csv"))
        with z.open(csv_name) as f:
            return pd.read_csv(f, usecols=COLS, engine="pyarrow")


for year in YEARS:
    zip_files = sorted(
        (DATA / str(year)).glob("*.zip"),
        key=lambda p: int(p.stem.rsplit("_", 1)[-1]),  # orden por mes: 1, 2, ..., 12
    )
    if not zip_files:
        print(f"{year}: no hay ZIPs, se omite")
        continue

    frames = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(read_zip, p): p for p in zip_files}
        for fut in as_completed(futures):
            p = futures[fut]
            try:
                frames[p] = fut.result()
                print(f"ok   {p.name} ({len(frames[p]):,} filas)")
            except Exception as e:
                print(f"FAIL {p.name}: {e}")

    df = pd.concat([frames[p] for p in zip_files if p in frames], ignore_index=True)
    out = DATA / f"bts_{year}_combined.parquet"
    df.to_parquet(out, index=False)
    print(f"{year}: {df.shape} -> {out}\n")
    del frames, df
