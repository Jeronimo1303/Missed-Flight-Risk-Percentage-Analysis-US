# Missed Flight Risk — US Domestic Flights

Predicting, **before departure**, the probability that a US domestic commercial flight leaves
15 minutes or more after its scheduled time (`DepDel15 = 1`).

Universidad EAFIT — Proyecto Integrador 1. Methodology: ASUM-DM.

**Team:** Jerónimo Acosta Acevedo, Santiago Gomez, Alejandro Quintero Moreno, Victor Daniel Delgado.

## Data prerequisite

Before starting the project's analysis, the raw BTS data (*Reporting Carrier On-Time Performance*,
2022-01 to 2026-08, about 32.4 M flights) was converted from the monthly `.zip` files to `.csv`
and then to Parquet, one file per month, partitioned by year and month and restricted to the 35
columns used in the project. This is the **bronze layer** of the Medallion architecture:

```
data/bronze/parquet/year=YYYY/month=MM/flights.parquet
```

The conversion script is `src/utils/csv_to_parquet.py`. The raw `.zip` files remain the original
copy of the data. The data itself is not versioned in this repository.

## Project structure

```
data/bronze/parquet/     # converted data (prerequisite above)
data/silver/flights/     # output of the EDA / leakage notebook
outputs_eda/             # figures and leakage audit
src/
├── flight_delay_eda_leakage.ipynb
└── utils/csv_to_parquet.py
```

## How to run

Open `src/flight_delay_eda_leakage.ipynb` and run it top to bottom. It reads the bronze layer,
audits the columns for data leakage, and writes the cleaned silver layer.

Requirements: Python 3.9+, `duckdb`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`.

## Data source

Bureau of Transportation Statistics. *Reporting carrier on-time performance (1987–present)*
[Data set]. U.S. Department of Transportation.
https://www.transtats.bts.gov/DL_SelectFields.asp?gnoyr_VQ=FGJ

Project Drive. 
https://drive.google.com/file/d/1cE-UlXny8HNnc7SW8mTAhpKbvGNIjZQc/view?usp=sharing