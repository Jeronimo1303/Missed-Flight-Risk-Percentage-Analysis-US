# Predicción de retrasos de vuelos BTS — Pipeline de datos y Machine Learning en AWS

Guía de **reproducción de extremo a extremo** del proyecto: desde la descarga de datos del *Bureau of Transportation Statistics* (BTS), su almacenamiento y procesamiento distribuido, hasta el entrenamiento de LightGBM, su despliegue en Amazon SageMaker y su consumo mediante una API HTTP.

> **Alcance de este README:** indica qué recursos crear en AWS, qué archivos utilizar y en qué orden ejecutarlos. **No incluye el código fuente de los scripts y notebooks:** cada archivo se incorpora por separado al repositorio.
>
> **Ambiente utilizado:** AWS Academy Learner Lab, rol `LabRole`, región `us-east-1`. Las cuentas AWS convencionales requieren roles y políticas IAM equivalentes. Algunas acciones pueden estar limitadas en Learner Lab.

## 1. Arquitectura general

**Ingesta:** BTS (ZIP mensual) → Amazon EventBridge Scheduler → AWS Step Functions → Lambda `bts-generate-periods` → Lambda `bts-ingestion` → Amazon S3 (`landing/raw/` y `bronze/parquet/`).

**Procesamiento:** Glue `bts-flights-bronze-to-silver` → S3 `silver/flights/`.

**Modelado:** SageMaker Notebook Instance → ingeniería de variables (DuckDB) → modelos base (LightGBM, XGBoost, regresión logística y K-means) → comparación → tuning con Optuna y reentrenamiento de LightGBM → Gold y artefactos en S3.

**Inferencia:** modelo versionado en S3 → SageMaker Model → Endpoint `bts-flight-delay-api` → Lambda `bts-inference-proxy` → Amazon API Gateway `POST /predict` → `enviar_prediccion_api.py`.

**Analítica (independiente de la API):** el dashboard de Streamlit consulta **directamente S3**, principalmente las tablas Silver/Gold. **No utiliza API Gateway ni solicita predicciones al endpoint.** Athena/Glue Data Catalog pueden utilizarse para consultas y validaciones, pero no forman parte de la ruta de consumo del dashboard.

El modelo predice **retraso en la salida de al menos 15 minutos** (`DepDel15`), con un horizonte de predicción de **240 minutos**; no predice directamente que un pasajero pierda un vuelo de conexión.

## 2. Requisitos y configuración previa

- Cuenta AWS con acceso a **S3, Lambda, Glue, Step Functions, SageMaker, API Gateway y CloudWatch**. EventBridge Scheduler es opcional si se desea dejar la ejecución mensual programada.
- En AWS Academy: iniciar Learner Lab y utilizar `LabRole` para los servicios donde la consola permita seleccionarlo. **No guardar claves de acceso del laboratorio en el repositorio.**
- Seleccionar **`us-east-1`** para todos los recursos.
- Python 3 para scripts locales; JupyterLab con las librerías usadas en los notebooks (`boto3`, `numpy`, `pandas`, `pyarrow`, `duckdb`, `scikit-learn`, `lightgbm`, `xgboost`, `optuna`, `matplotlib`, `joblib`, etc.). La preparación exacta del entorno está en las celdas de dependencias de los notebooks.
- Para consumir la API desde el equipo local: `requests` y `python-dotenv`.
- Capacidad de almacenamiento persistente suficiente en la Notebook Instance para descargar Silver y guardar los checkpoints de DuckDB y Parquet; ajustar el volumen EBS según disponibilidad y espacio libre.

### Identificadores utilizados

| Recurso | Nombre o valor |
|---|---|
| Región | `us-east-1` |
| Bucket S3 | `bts-flights-analysis` |
| Rol IAM en AWS Academy | `LabRole` |
| Lambda de períodos | `bts-generate-periods` |
| Lambda de ingesta | `bts-ingestion` |
| Glue ETL | `bts-flights-bronze-to-silver` |
| Step Functions | `bts-flights-pipeline` |
| Endpoint SageMaker | `bts-flight-delay-api` |
| Lambda proxy de inferencia | `bts-inference-proxy` |
| HTTP API | `bts-flight-delay-http-api` |
| Ruta HTTP | `POST /predict` |

Si se usa otro bucket (los nombres S3 son globalmente únicos), actualizar **todas** las constantes `BUCKET`/`BUCKET_NAME`, los parámetros del job, la State Machine y los scripts que referencian el nombre original.

## 3. Archivos que deben incluirse en el repositorio

La organización siguiente es **sugerida**; las rutas de ejecución de los notebooks suponen que estos se cargan en `/home/ec2-user/SageMaker/src/`.

| Grupo | Archivos |
|---|---|
| Ingesta | `bts_backfill_raw_historico.py`, `bts-generate-periods-lambda-auto.zip`, `bts-ingestion-lambda-corregida.zip` |
| Orquestación | `bts-step-functions-ingesta-a-silver.json` |
| ETL | `bts_bronze_to_silver.py` (script del job Glue ya utilizado) |
| Utilidades del modelado | `model_features.py`, `model_eval.py` |
| Etapa A | `flight_delay_modeling_etapa_A_aws.ipynb` |
| Modelos base | `flight_delay_modeling_etapa_B1_lightgbm_float64_aws.ipynb`, `flight_delay_modeling_etapa_B2_logreg_aws.ipynb`, `flight_delay_modeling_etapa_B3_xgboost_float64_aws.ipynb`, `flight_delay_modeling_etapa_B4_kmeans_aws.ipynb` |
| Análisis C | `flight_delay_analisis_etapa_C_definitivo_meta_corregido_aws.ipynb` |
| Experimentos adicionales (opcionales) | `flight_delay_etapa_D1_experimentos_lightgbm_float64_aws.ipynb`, `flight_delay_etapa_D2_diagnostico_deriva_float64_aws.ipynb` |
| Tuning y modelo final | `flight_delay_tuning_etapa_E_float64_sagemaker_meta_corregido.ipynb` |
| Despliegue SageMaker | `flight_delay_etapa_F_despliegue_endpoint_sagemaker.ipynb`, `bts_sagemaker_endpoint.py` |
| Inferencia HTTP | `bts-inference-proxy-lambda.zip`, `enviar_prediccion_api.py`, `.env.example` |

**Importante:** no utilizar los notebooks antiguos `float32`, el notebook monolítico inicial ni las versiones anteriores al ajuste de `meta.json` para reproducir la versión final. Los modelos LightGBM y XGBoost base se evalúan con entradas `float64`; los otros dos modelos mantienen sus configuraciones originales.

---

## 4. Crear el Data Lake en Amazon S3

1. AWS Console → **S3 → Create bucket**.
2. Nombre: `bts-flights-analysis`; región `us-east-1`.
3. Mantener **Block all public access** habilitado y usar cifrado administrado de S3.
4. Preparar los siguientes prefijos. En S3 no es obligatorio crearlos como carpetas: se materializan cuando se suben objetos.

| Prefijo | Contenido |
|---|---|
| `landing/raw/` | ZIP originales de BTS |
| `bronze/parquet/` | Parquet mensual producido por Lambda o cargado históricamente |
| `silver/flights/` | Datos transformados y particionados por Glue |
| `control/manifests/` | Manifiestos de la ingesta automática |
| `gold/pred_vs_real/` | Tablas analíticas del modelo final |
| `modeling/checkpoints/` | Resultados de modelos base |
| `modeling/outputs_modeling/` | Comparación de modelos y otros análisis |
| `modeling/tuning/` | Trials de Optuna y métricas del tuning |
| `models/flight-delay/` | Paquetes versionados del modelo final |
| `modeling/deployment/` | Evidencia de inferencia del endpoint |

Cobertura histórica utilizada en el proyecto: **enero de 2022 a agosto de 2026 (56 meses)**. Septiembre de 2026 se incorporó posteriormente como prueba de ingesta automatizada.

## 5. Ingestar datos BTS: dos formas de reproducir el histórico

La fuente es el conjunto mensual **Reporting Carrier On-Time Performance** de BTS. Cada ZIP contiene un CSV con los registros del mes.

### 5.1. Reproducción desde cero — ruta automatizada

Si el bucket está vacío, utilizar las Lambdas y Step Functions de los apartados **6 a 8** para procesar los meses históricos. La Lambda de ingesta **hace las dos operaciones**: conserva el ZIP en `landing/raw/` y convierte CSV → Parquet en `bronze/parquet/`.

Se recomienda procesar el histórico **por bloques de meses**, verificando los resultados entre bloques, en lugar de iniciar una única ejecución extensa. Después se ejecuta Glue para producir Silver. **No hace falta un job adicional Raw → Bronze**.

### 5.2. Procedimiento utilizado inicialmente en este proyecto — Bronze previo + backfill Raw

En el desarrollo original, los Parquet Bronze históricos se prepararon y cargaron inicialmente de manera manual. Luego se completó Raw con los ZIP originales de BTS utilizando `bts_backfill_raw_historico.py`, sin reescribir Bronze.

Ejecutar desde la terminal de SageMaker, con credenciales de `LabRole` vigentes:

```bash
python bts_backfill_raw_historico.py --start 2022-01 --end 2026-08
python bts_backfill_raw_historico.py --start 2022-01 --end 2026-09 --check-only
```

El script es reanudable y omite los archivos que ya existen. La estructura resultante es `landing/raw/year=YYYY/month=MM/flights.zip`.

**Advertencia:** el backfill Raw **no crea manifiestos de la Lambda**. No repetir los 56 meses con la Lambda sobre ese Bronze existente sin planificar la posible reconversión y sobrescritura de Parquet. Presentar en la documentación la carga histórica y la ingesta automatizada como fases distintas.

## 6. Crear las Lambdas de ingesta

AWS Console → **Lambda → Create function → Author from scratch**. En ambas utilizar **Python 3.12**, arquitectura **x86_64**, rol de ejecución existente **`LabRole`** y handler **`lambda_function.lambda_handler`**.

### 6.1. `bts-generate-periods`

1. Crear la función con nombre `bts-generate-periods`.
2. Configurar **128 MB de memoria** y **30 segundos de timeout**.
3. Code → Upload from → **.zip file** → cargar `bts-generate-periods-lambda-auto.zip` → **Deploy**.
4. No necesita layers ni variables de entorno.
5. Probar desde **Test** con un evento que contenga `start: "2026-09"` y `end: "2026-09"`; verificar que devuelva una lista con un período.
6. También admite `mode: "monthly"` y `lag_months: 2` para seleccionar automáticamente el mes con dos meses de rezago.

### 6.2. `bts-ingestion`

1. Crear la función con nombre `bts-ingestion`.
2. Configurar **3072 MB de memoria**, **900 segundos de timeout** y **2048 MB de almacenamiento temporal `/tmp`**.
3. Code → Upload from → **.zip file** → cargar `bts-ingestion-lambda-corregida.zip` → **Deploy**.
4. Configuration → Environment variables: `BUCKET_NAME=bts-flights-analysis`.
5. **Agregar la layer pública de AWS SDK for pandas (incluye PyArrow)**: Code → Layers → Add a layer → Specify an ARN, con:

   `arn:aws:lambda:us-east-1:336392948345:layer:AWSSDKPandas-Python312:32`

6. Verificar que coincidan el runtime `python3.12` y la arquitectura `x86_64`.
7. Test: enviar un evento con `year: 2026` y `month: 9`.
8. Comprobar en S3 que se generen:
   - `landing/raw/year=2026/month=09/flights.zip`
   - `bronze/parquet/year=2026/month=09/flights.parquet`
   - `control/manifests/year=2026/month=09/ingestion.json`
9. El manifiesto debe indicar `SUCCESS`; una segunda ejecución correcta debería responder `SKIPPED`.

La función verifica ZIP, columnas, Parquet y tamaños, y registra metadatos de la ingesta. Revisar errores en **CloudWatch Logs**. La Lambda requiere conectividad HTTP/HTTPS con BTS; evitar una VPC sin salida a Internet.

## 7. Crear el job de procesamiento distribuido en Glue

**No se crea otro job para Raw → Bronze**: esa conversión la hace `bts-ingestion`.

1. AWS Console → **AWS Glue → ETL jobs → Script editor** → motor **Spark**.
2. Pegar/subir el script existente `bts_bronze_to_silver.py`.
3. Guardar con nombre **`bts-flights-bronze-to-silver`**.
4. Seleccionar el rol permitido (`LabRole` en el laboratorio), **Glue 4.0**, Python 3, workers **G.1X**. Ajustar de 2 a 10 workers según cuota y presupuesto.
5. Configurar los argumentos del job:
   - `--BUCKET`: `bts-flights-analysis`
   - `--EXPECTED_START`: `2022-01`
   - `--EXPECTED_END`: el último mes que se desea procesar (inicialmente `2026-08`, luego `2026-09`, etc.)
   - `--MIN_ROWS_PER_MONTH`: usar el valor definido por el script y la fuente; en la configuración inicial se contempló `300000`.
   - `--FORCE`: `false` para ejecución incremental; utilizar `true` solo si es necesario reprocesar.
6. **Save → Run**. Verificar en CloudWatch el estado del job.
7. Comprobar en S3 el resultado en `silver/flights/Year=YYYY/Month=M/`.

Este es el **procesamiento distribuido PySpark** del proyecto: limpia y valida los datos Bronze, aplica tipados y produce Silver para análisis y modelado.

## 8. Crear la State Machine y programar la ingesta

### 8.1. AWS Step Functions

1. Console → **Step Functions → Create state machine → Blank**, tipo **Standard**.
2. Nombre: **`bts-flights-pipeline`**.
3. Elegir el rol existente **`LabRole`** si la relación de confianza y los permisos del laboratorio lo permiten.
4. En el editor de código, cargar el contenido de **`bts-step-functions-ingesta-a-silver.json`**.
5. La State Machine ejecuta, en orden: **GeneratePeriods → IngestPeriods (Map) → BronzeToSilver (Glue `.sync`)**.
6. La versión entregada usa hasta **2 meses simultáneos** (`MaxConcurrency: 2`) y espera a que Glue termine.
7. **Start execution** con `start=2026-09` y `end=2026-09`; verificar que todos los estados terminen correctamente.
8. Revisar el nuevo mes en S3 Silver, además de Raw/Bronze y el manifiesto.

La State Machine pasa el valor `end` al parámetro `--EXPECTED_END` de Glue; mantener actualizado `--EXPECTED_START` según el histórico completo que procesa ese job. Para el histórico desde cero, ejecutar por rangos y revisar el resultado entre bloques.

### 8.2. Amazon EventBridge Scheduler (programación opcional)

1. **EventBridge Scheduler → Create schedule → Recurring**.
2. Nombre sugerido: `bts-monthly-ingestion`.
3. Frecuencia: mensual, por ejemplo el **día 22 a las 09:00 en `America/Bogota`**.
4. Target: **Step Functions**, máquina `bts-flights-pipeline`.
5. Input: `mode=monthly` y `lag_months=2`.
6. Configurar un rol asumible por Scheduler con permiso `states:StartExecution`.
7. Validar una ejecución real y monitorear fallos en Step Functions/CloudWatch.

**Restricción de AWS Academy:** `LabRole` puede no tener una relación de confianza válida para `scheduler.amazonaws.com`, o la cuenta puede bloquear la creación de roles. En tal caso, la reproducción **manual** desde **Start execution** sigue siendo válida; no afirmar que Scheduler quedó activo si no se comprobó.

La disponibilidad de archivos BTS puede sufrir retrasos variables; un rezago de dos meses no garantiza que el ZIP exista. Verificar la publicación y configurar seguimiento de fallos.

---

## 9. Crear y preparar SageMaker Notebook Instance

1. **Amazon SageMaker AI → Notebook instances → Create notebook instance**.
2. Nombre sugerido: `bts-modeling`; rol **`LabRole`**.
3. Instancia de trabajo utilizada: **`ml.t3.xlarge`** (4 vCPU, 16 GiB RAM). Seleccionar capacidad de disco persistente suficiente para Silver y los checkpoints.
4. Cuando esté **InService**, abrir **JupyterLab**.
5. Crear y trabajar dentro de **`/home/ec2-user/SageMaker/src/`**. Esto es importante: los notebooks utilizan rutas relativas `../data/` y `../outputs/`.
6. Cargar los notebooks de etapas A–F y `bts_sagemaker_endpoint.py` en `src/`; guardar `model_features.py` y `model_eval.py` dentro de `src/utils/` para los notebooks que los importan.
7. Instalar las dependencias indicadas en los notebooks, seleccionar un kernel que realmente contenga dichas librerías y reiniciarlo si fuera necesario.
8. Descargar una copia de Silver desde S3 al disco persistente de SageMaker, respetando el árbol de subdirectorios:

```bash
cd /home/ec2-user/SageMaker/src
aws s3 sync s3://bts-flights-analysis/silver/flights/ ../data/silver/flights/
```

Verificar que existan Parquet en `../data/silver/flights/Year=YYYY/Month=M/` antes de empezar. En un Learner Lab con permisos renovables, comprobar `aws sts get-caller-identity` al iniciar una nueva sesión.

## 10. Ingeniería de variables y entrenamiento de los modelos originales

**Ejecutar los notebooks en este orden, desde `SageMaker/src/`.** No utilizar `Run All` sobre toda la colección: cada etapa tiene checkpoints propios y carga de memoria significativa.

| Orden | Notebook | Acción y resultado |
|---|---|---|
| **A** | `flight_delay_modeling_etapa_A_aws.ipynb` | Construir características históricas y operativas con DuckDB, evitando información futura; crear Parquet particionados en `../data/checkpoints/features_seed42_h240_chunk50000/` (4 M train, 1 M val y 1 M test). **Solo una vez**. |
| **B1** | `flight_delay_modeling_etapa_B1_lightgbm_float64_aws.ipynb` | Entrenar el LightGBM original con variables numéricas `float64`; guardar modelo, métricas y predicciones en `checkpoints/lightgbm_float64/` y respaldar en S3. |
| **B2** | `flight_delay_modeling_etapa_B2_logreg_aws.ipynb` | Entrenar regresión logística original; guardar modelo, predicciones y métricas. |
| **B3** | `flight_delay_modeling_etapa_B3_xgboost_float64_aws.ipynb` | Entrenar XGBoost con el esquema/categorías del **B1 float64**; guardar modelo, métricas y predicciones. |
| **B4** | `flight_delay_modeling_etapa_B4_kmeans_aws.ipynb` | Entrenar K-means y generar probabilidades por clúster para la comparación. |
| **C** | `flight_delay_analisis_etapa_C_definitivo_meta_corregido_aws.ipynb` | Ejecutar la **Parte I**: PR-AUC, ROC-AUC, umbrales, matrices de confusión, bootstrap, curvas, calibración, subgrupos, tiempos y decisión de modelo para tuning. Publicar resultados de C en S3. |

**Resultados de las etapas B/C:**

- `s3://bts-flights-analysis/modeling/checkpoints/`
- `s3://bts-flights-analysis/modeling/outputs_modeling/comparison_aws_seed42_h240/`

La elección para optimizar **LightGBM** se justificó por relación entre calidad predictiva y tiempo de entrenamiento. **XGBoost tuvo un PR-AUC ligeramente superior en la comparación base**; no alterar esa conclusión. La selección definitiva del artefacto desplegable ocurre después del tuning.

### Experimentos adicionales anteriores al tuning (opcionales)

Si se quiere reproducir toda la evidencia experimental, ejecutar también:

1. `flight_delay_etapa_D1_experimentos_lightgbm_float64_aws.ipynb`: ablación de grupos de variables y ponderación de clases.
2. `flight_delay_etapa_D2_diagnostico_deriva_float64_aws.ipynb`: diagnóstico de deriva temporal.

Se apoyan en los checkpoints existentes. **No son prerrequisitos técnicos para crear el endpoint**.

## 11. Tuning LightGBM, reentrenamiento, Gold y publicación

Abrir **`flight_delay_tuning_etapa_E_float64_sagemaker_meta_corregido.ipynb`**.

Ejecutar sus secciones **en orden**, verificando checkpoints entre etapas:

1. **Preparación**: confirmar los Parquet de A, el esquema y la referencia B1 float64.
2. **Optuna**: ejecutar la celda de preparación del estudio y **después la celda de búsqueda**. Son celdas distintas; el mensaje `Trials ya registrados: 1 / 40` no significa que el entrenamiento haya comenzado.
3. **Búsqueda**: hasta **40 trials**, utilizando 1 M train / 500 mil val como muestra proxy. El estado se guarda en SQLite; los trials `PRUNED` son podas válidas, no errores.
4. **Reanudación**: si vence el tiempo o termina el Learner Lab, renovar credenciales y volver a ejecutar la búsqueda; no borrar `study.sqlite3` ni iniciar un estudio nuevo.
5. **Confirmación**: ejecutar los tres mejores candidatos con 4 M train / 1 M val; seleccionar por validación, comparando también con B1.
6. **Prueba y refit**: evaluar una vez en test y reentrenar el modelo elegido con train + val (5 M registros).
7. **Empaquetado**: ejecutar la sección del contrato de inferencia, que genera un **`model.tar.gz` versionado**, `feature_schema.json`, `metadata.json`, `sample_request.json` y `inference.py` dentro del paquete.
8. **Gold definitivo**: generar las cuatro tablas de predicción frente a realidad: `group_metrics`, `score_bins`, `timeseries` y `heat_hora_dia`; revisar `meta.json` con la **versión real del modelo**.
9. **Publicación S3**: ejecutar la última celda autónoma de subida. Puede correrse tras reiniciar el laboratorio **sin reentrenar**, siempre que los archivos finales sigan en el almacenamiento persistente.

Las salidas principales quedan en:

- Local: `../outputs/outputs_modeling/tuning_lightgbm_float64/`
- Local: `../outputs/model_registry/<model_version>/`
- S3: `modeling/tuning/lightgbm-float64-seed42-h240/`
- S3: `models/flight-delay/<model_version>/model.tar.gz`
- S3: `gold/pred_vs_real/` (4 tablas + `meta.json`)

**Referencia de la ejecución documentada:** `lgbm-f64-seed42-h240-20261122T031926Z-trial-038` (LightGBM `4.7.0`, 37 variables, 7 categóricas, horizonte 240 min y umbral 0,25). Una reproducción puede obtener otra versión y otros hiperparámetros: utilizar siempre el `model_version.json` generado por **su** ejecución.

### Análisis Gold opcional

Volver a `flight_delay_analisis_etapa_C_definitivo_meta_corregido_aws.ipynb`, activar `RUN_GOLD_ANALYSIS = True` en la **Parte II** y ejecutar sus celdas para producir gráficas sobre las tablas Gold definitivas. El dashboard analítico puede leer esos Parquet **directamente desde S3**; no requiere API Gateway ni un endpoint de inferencia.

## 12. Desplegar LightGBM como SageMaker Endpoint

**Archivos:** `bts_sagemaker_endpoint.py` y, opcionalmente, `flight_delay_etapa_F_despliegue_endpoint_sagemaker.ipynb`.

El script lee el `model.tar.gz` **ya publicado en S3** y crea tres recursos separados: **SageMaker Model**, **Endpoint Configuration** y **Endpoint**. No reentrena el modelo.

Desde la terminal de `SageMaker/src/`:

```bash
python bts_sagemaker_endpoint.py preflight
python bts_sagemaker_endpoint.py deploy --instance ml.m5.large
python bts_sagemaker_endpoint.py status
python bts_sagemaker_endpoint.py invoke
```

- `preflight`: comprueba S3, integridad del artefacto, versión y características.
- `deploy`: crea o reutiliza el Model y EndpointConfig, y solicita el endpoint `bts-flight-delay-api`.
- `status`: comprobar repetidamente hasta ver **`InService`**.
- `invoke`: probar con `sample_request.json`, comparar la probabilidad con la prueba del tuning y almacenar evidencia en S3.

**Corrección importante:** la celda original de la etapa F y el valor predeterminado del script indican `ml.t3.medium`, pero **ese tipo falló en `CreateEndpointConfig` por no ser admitido para este endpoint**. Utilizar expresamente **`--instance ml.m5.large`**; si se usa el notebook F, cambiar también su celda `deploy` a ese tipo. La disponibilidad final depende de la cuota de la cuenta AWS.

El contenedor administrado empleado es SageMaker Scikit-learn `1.9-0`; el paquete contiene los requisitos de LightGBM y el código de inferencia personalizado. Revisar CloudWatch si falla el arranque o la instalación de dependencias.

**Evidencia de la invocación:**

`modeling/deployment/flight-delay/<model_version>/endpoint_test_response.json`

El endpoint recibe **las 37 características ya calculadas**. No acepta un CSV BTS sin procesar ni calcula automáticamente las variables históricas de T−240 min.

## 13. Exponer el endpoint mediante API Gateway y Lambda

El endpoint SageMaker es privado. La aplicación externa no lo llama directamente: se usa una **HTTP API** con una Lambda intermediaria que invoca SageMaker mediante `LabRole`.

### 13.1. Crear `bts-inference-proxy`

1. AWS Lambda → **Create function → Author from scratch**.
2. Nombre: **`bts-inference-proxy`**; runtime **Python 3.12**; arquitectura **x86_64**; rol **`LabRole`**.
3. Memoria **256 MB**, timeout **25 segundos**.
4. Cargar `bts-inference-proxy-lambda.zip` mediante **Upload from → .zip file**.
5. Handler `lambda_function.lambda_handler`. **No requiere layer**.
6. Variables de entorno:
   - `SAGEMAKER_ENDPOINT=bts-flight-delay-api`
   - `API_SHARED_TOKEN=<token-secreto-generado-localmente>`
7. Probar primero una solicitud **sin token**: la respuesta debe ser **HTTP 401**.
8. Verificar que `LabRole` pueda efectuar `sagemaker:InvokeEndpoint` y que la Lambda escriba logs en CloudWatch.

La capa PyArrow se utiliza en la Lambda **de ingesta**, **no** en esta Lambda de inferencia.

### 13.2. Crear HTTP API

1. **API Gateway → Create API → HTTP API → Build**.
2. Nombre: **`bts-flight-delay-http-api`**.
3. Integración: Lambda **`bts-inference-proxy`**.
4. Ruta: **`POST /predict`**.
5. Stage: **`$default`**, con **Auto-deploy**.
6. La Lambda valida `Authorization: Bearer <token>` antes de llamar a SageMaker; no se requiere firma IAM desde el equipo cliente.
7. Verificar permiso de invocación **API Gateway → Lambda** (`lambda:InvokeFunction`) y activar monitoreo/logs. Configurar *throttling* para limitar solicitudes.
8. Copiar la **Invoke URL real** del stage en la consola y añadir `/predict`. No hardcodear un identificador inventado.

Esta protección mediante token compartido es adecuada para una demostración **controlada**. Para una API pública de producción se recomienda autenticación de usuarios robusta, controles antiabuso y límites de consumo.

## 14. Consumir la API desde un script de Python

Archivo: **`enviar_prediccion_api.py`**. No depende de credenciales temporales de AWS Academy.

1. En el equipo cliente, instalar dependencias:

   ```bash
   pip install requests python-dotenv
   ```

2. Copiar `.env.example` como `.env` junto al script.
3. Completar `API_GATEWAY_URL` con la URL HTTPS **completa** terminada en `/predict` y `API_BEARER_TOKEN` con el mismo token de `API_SHARED_TOKEN` de Lambda.
4. Copiar desde los artefactos del tuning `sample_request.json` (contiene las 37 variables y valores de un ejemplo válido).
5. Ejecutar:

   ```bash
   python enviar_prediccion_api.py sample_request.json
   python enviar_prediccion_api.py sample_request.json --output resultado.json
   ```

6. Verificar respuesta HTTP 200, `model_version`, `probability_delay_15min`, `alert` y `decision_threshold`.

**Formatos de entrada admitidos:** objeto `features` para un vuelo u objeto `instances` con una lista de **1 a 100 vuelos**. Cada vuelo necesita las **37 características del esquema**, no solo aeropuerto y hora. `DepDel15` es el objetivo y **no** se envía.

Para el ejemplo de despliegue guardado, la ejecución documentada devolvió aproximadamente **0,12326939** de probabilidad y `alert=false`, con umbral **0,25**. En una reproducción, la respuesta dependerá del modelo versionado y de la observación utilizada.

**Seguridad:** `.env` y cualquier archivo con credenciales o tokens reales deben estar en `.gitignore`; publicar únicamente `.env.example`.

---

## 15. Lista de verificación de reproducción

| Fase | Evidencia de éxito |
|---|---|
| S3 | Bucket creado; prefijos Raw, Bronze, Silver, Gold y Models disponibles cuando se cargan objetos |
| Raw histórico | ZIP en `landing/raw/year=YYYY/month=MM/flights.zip` para todo el rango esperado |
| Lambda de ingesta | Manifiesto `SUCCESS` y Parquet en `bronze/parquet/`; segunda ejecución `SKIPPED` |
| Step Functions | `GeneratePeriods`, `IngestPeriods` y `BronzeToSilver` completados |
| Glue | Job completado y datos de Silver disponibles para todos los meses procesados |
| Etapa A | 4 M train, 1 M validación y 1 M prueba persistidos en Parquet |
| B1–B4 / C | Cuatro checkpoints y tablas/gráficas de comparación publicados en S3 |
| Optuna / refit | 40 trials contabilizados, candidato seleccionado, `model.tar.gz` y contrato de inferencia |
| Gold | `group_metrics`, `score_bins`, `timeseries`, `heat_hora_dia` y `meta.json` publicados |
| SageMaker | Endpoint `bts-flight-delay-api` en estado **`InService`** |
| Inferencia SageMaker | Probabilidad del endpoint coincide con la de prueba del tuning |
| API Gateway | `POST /predict` rechaza solicitudes sin token (`401`) y acepta una solicitud válida (`200`) |
| Cliente Python | `enviar_prediccion_api.py` lee `.env` y devuelve la respuesta JSON |

## 16. Recuperación de errores y costos

- **Caducidad de credenciales (`NoCredentialsError`)**: reiniciar/renovar Learner Lab y volver a ejecutar **solo las celdas de publicación o los scripts de estado**. Los Parquet y modelos ya guardados en el volumen persistente de SageMaker no necesitan reentrenarse.
- **Optuna interrumpido**: conservar `study.sqlite3`; los trials finalizados y podados se mantienen para reanudar.
- **`CreateEndpointConfig` con `ml.t3.medium`**: usar `ml.m5.large`, no recrear innecesariamente el SageMaker Model ya generado.
- **Endpoint en `Failed`**: revisar *Failure reason* en SageMaker y logs del contenedor en CloudWatch.
- **API 401**: verificar el mismo token en `.env` y en `API_SHARED_TOKEN` de Lambda.
- **API 5xx**: revisar permisos API Gateway → Lambda, permisos Lambda → SageMaker, estado del endpoint y logs CloudWatch.
- **Datos de predicción inválidos**: utilizar `feature_schema.json` y `sample_request.json`; el endpoint no sustituye la ingeniería de variables.
- **Costo**: detener la Notebook Instance cuando no se use. **Detener la notebook no detiene el endpoint**; la inferencia en tiempo real sigue aprovisionada y puede generar cargos.

### Eliminar el endpoint al finalizar la demostración

Desde SageMaker, y **solo cuando se haya completado la demostración**:

```bash
python bts_sagemaker_endpoint.py cleanup --confirm
```

Este comando elimina el endpoint, su configuración y el objeto SageMaker Model correspondiente, pero **conserva los artefactos versionados de S3**. Revisar aparte las Lambdas, API Gateway, Scheduler y otros recursos que ya no se requieran.

## 17. Alcance y reproducibilidad

- La ingesta de **nuevos meses** está diseñada para automatizarse; el histórico de este desarrollo se cargó inicialmente mediante el proceso manual Bronze más el posterior backfill de Raw. Una reproducción *desde cero* puede utilizar las Lambdas para ambos pasos.
- Glue realiza el ETL distribuido; **el entrenamiento en SageMaker Notebook es de un solo nodo**, usando muestras y checkpoints. No presentar el entrenamiento como Spark distribuido.
- Los resultados del modelo son reproducibles con los mismos datos, semillas, particiones y versiones de dependencias, aunque no se garantiza igualdad bit a bit entre máquinas o versiones.
- El dashboard Streamlit es **analítico** y consulta S3 directamente; la API HTTP está destinada a **scripts Python de inferencia**, no al dashboard.
- SageMaker predice retraso de salida de al menos 15 minutos y requiere variables previamente calculadas a T−240 minutos; no recibe directamente un vuelo crudo desde BTS.
- **No guardar en GitHub** `.env`, claves IAM, contraseñas, tokens, datos sensibles ni grandes archivos de entrenamiento que corresponden a S3.

### Documentación oficial útil

- [AWS Step Functions + Glue (`.sync`)](https://docs.aws.amazon.com/step-functions/latest/dg/connect-glue.html)
- [Layer AWS SDK for pandas en Lambda](https://aws-sdk-pandas.readthedocs.io/en/stable/layers.html)
- [Desplegar modelos en endpoints SageMaker](https://docs.aws.amazon.com/sagemaker/latest/dg/realtime-endpoints-deploy-models.html)
- [API Gateway HTTP API con Lambda proxy](https://docs.aws.amazon.com/apigateway/latest/developerguide/http-api-develop-integrations-lambda.html)
