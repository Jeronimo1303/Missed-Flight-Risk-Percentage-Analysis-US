CREATE DATABASE IF NOT EXISTS bts_flights;

CREATE EXTERNAL TABLE IF NOT EXISTS bts_flights.gold_group_metrics (
  `dimension` string,
  `grupo` string,
  `vuelos` bigint,
  `retrasados` bigint,
  `tasa_real` double,
  `prob_media` double,
  `sesgo` double,
  `roc_auc` double,
  `pr_auc` double,
  `brier` double,
  `umbral` double,
  `alertas` bigint,
  `tp` bigint,
  `fp` bigint,
  `fn` bigint,
  `tn` bigint,
  `precision` double,
  `recall` double,
  `f1` double,
  `orden` double,
  `conjunto` string
)
STORED AS PARQUET
LOCATION 's3://bts-flights-analysis/gold/pred_vs_real/group_metrics/';

CREATE EXTERNAL TABLE IF NOT EXISTS bts_flights.gold_score_bins (
  `bin` bigint,
  `p_min` double,
  `p_max` double,
  `p_media` double,
  `vuelos` bigint,
  `retrasados` BIGINT,
  `conjunto` string
)
STORED AS PARQUET
LOCATION 's3://bts-flights-analysis/gold/pred_vs_real/score_bins/';

CREATE EXTERNAL TABLE IF NOT EXISTS bts_flights.gold_timeseries (
  `fecha` timestamp,
  `vuelos` bigint,
  `retrasados` BIGINT,
  `prob_suma` double,
  `alertas` bigint,
  `tp` bigint,
  `fp` bigint,
  `conjunto` string
)
STORED AS PARQUET
LOCATION 's3://bts-flights-analysis/gold/pred_vs_real/timeseries/';

CREATE EXTERNAL TABLE IF NOT EXISTS bts_flights.gold_heat_hora_dia (
  `hora` bigint,
  `dia_semana` bigint,
  `vuelos` bigint,
  `retrasados` BIGINT,
  `prob_suma` double,
  `conjunto` string
)
STORED AS PARQUET
LOCATION 's3://bts-flights-analysis/gold/pred_vs_real/heat_hora_dia/';
