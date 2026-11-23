#!/usr/bin/env python3
"""Despliegue y prueba (reanudables) de LightGBM BTS en SageMaker AI.

No entrena ni recalcula variables históricas. Python 3 + boto3.
Ejecutar dentro de SageMaker/src con permisos del LabRole.
"""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import re
import tarfile
import tempfile
import time
from datetime import datetime, timezone

import boto3
from botocore.exceptions import ClientError

REGION = "us-east-1"
BUCKET = "bts-flights-analysis"
TUNING_PREFIX = "modeling/tuning/lightgbm-float64-seed42-h240/"
ENDPOINT_NAME = "bts-flight-delay-api"
IMAGE_URI = "683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-scikit-learn:1.9-0"
DEFAULT_INSTANCE = "ml.t3.medium"   # Si la cuenta no lo permite, probar ml.m5.large.
LOCAL_EVIDENCE = Path("../outputs/outputs_modeling/deployment_sagemaker")


def aws():
    sess = boto3.Session(region_name=REGION)
    if sess.get_credentials() is None:
        raise RuntimeError("No hay credenciales AWS. Inicia/reinicia Learner Lab.")
    return sess


def _missing(exc):
    code = exc.response.get("Error", {}).get("Code", "")
    msg = exc.response.get("Error", {}).get("Message", "")
    return code in ("ResourceNotFound", "ResourceNotFoundException", "404", "NoSuchKey", "NotFound") or (
        code == "ValidationException" and ("Could not find" in msg or "does not exist" in msg or "not found" in msg.lower())
    )


def load_info(s3):
    local = Path("../outputs/outputs_modeling/tuning_lightgbm_float64/model_version.json")
    try:
        body = s3.get_object(Bucket=BUCKET, Key=TUNING_PREFIX + "model_version.json")["Body"].read()
        info = json.loads(body)
        where = f"s3://{BUCKET}/{TUNING_PREFIX}model_version.json"
    except ClientError as e:
        if not _missing(e):
            raise
        if not local.is_file():
            raise RuntimeError("No existe model_version.json en S3 ni en disco. Ejecuta la sección 11 del tuning.") from e
        info = json.loads(local.read_text())
        where = str(local)
    version = info["model_version"]
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,160}", version):
        raise ValueError("model_version inválida")
    key = f"models/flight-delay/{version}/model.tar.gz"
    response = s3.head_object(Bucket=BUCKET, Key=key)
    assert response["ContentLength"] > 0, "model.tar.gz está vacío"
    digest = hashlib.sha1(version.encode()).hexdigest()[:10]
    return {
        "version": version,
        "info_source": where,
        "info": info,
        "key": key,
        "uri": f"s3://{BUCKET}/{key}",
        "model_name": f"bts-delay-model-{digest}",
        "config_name": f"bts-delay-config-{digest}",
        "endpoint_name": ENDPOINT_NAME,
        "bytes": response["ContentLength"],
    }


def _load_from_tar(tar, name):
    f = tar.extractfile(name)
    if f is None:
        raise ValueError(f"Falta {name} en model.tar.gz")
    return f.read()


def preflight(session, s3, info):
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "model.tar.gz"
        s3.download_file(BUCKET, info["key"], str(path))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        expected = info["info"].get("archive_sha256")
        if expected and expected != digest:
            raise ValueError(f"SHA256 incorrecto del modelo (S3 {digest} vs registrado {expected})")
        with tarfile.open(path, mode="r:gz") as tar:
            required = {"model.txt", "feature_schema.json", "metadata.json", "code/inference.py", "code/requirements.txt"}
            missing = required - set(tar.getnames())
            assert not missing, f"Archivo incompleto: {missing}"
            schema = json.loads(_load_from_tar(tar, "feature_schema.json"))
            metadata = json.loads(_load_from_tar(tar, "metadata.json"))
            requirements = _load_from_tar(tar, "code/requirements.txt").decode().strip()
            assert metadata["model_version"] == info["version"]
            assert metadata["prediction_horizon_minutes"] == 240
            assert schema["feature_names"] and schema["cat_cols"]
            assert "lightgbm==" in requirements.lower()
    print("\n✔ S3:", info["uri"])
    print("✔ Artefacto MiB:", round(info["bytes"] / 2**20, 2))
    print("✔ SHA-256:", digest)
    print("✔ Versión:", info["version"])
    print("✔ Variables:", len(schema["feature_names"]), "| Categóricas:", len(schema["cat_cols"]))
    print("✔ Dependencia:", requirements)
    print("✔ Umbral:", metadata["decision_threshold"])
    print("✔ Target:", metadata["target"])
    print("✔ Imagen administrada:", IMAGE_URI)
    print("✔ El endpoint espera características YA calculadas (horizonte 240 min).")
    return metadata


def role_arn(session):
    account = session.client("sts").get_caller_identity()["Account"]
    return f"arn:aws:iam::{account}:role/LabRole"


def describe_or_none(client, method, arg, val):
    try:
        return getattr(client, method)(**{arg: val})
    except ClientError as e:
        if _missing(e):
            return None
        raise


def deploy(session, info, instance):
    sm = session.client("sagemaker")
    role = role_arn(session)
    print("Versión del modelo:", info["version"])
    print("Rol de ejecución:", role)
    print("Tipo de instancia:", instance)

    present = describe_or_none(sm, "describe_model", "ModelName", info["model_name"])
    if present is None:
        print("Creando SageMaker Model:", info["model_name"])
        sm.create_model(
            ModelName=info["model_name"],
            ExecutionRoleArn=role,
            PrimaryContainer={
                "Image": IMAGE_URI,
                "ModelDataUrl": info["uri"],
                "Environment": {
                    "SAGEMAKER_PROGRAM": "inference.py",
                    "SAGEMAKER_SUBMIT_DIRECTORY": "/opt/ml/model/code",
                    "SAGEMAKER_CONTAINER_LOG_LEVEL": "20",
                    "OMP_NUM_THREADS": "1",
                },
            },
            Tags=[{"Key": "Project", "Value": "bts-flight-delay"}],
        )
        print("✔ Modelo creado")
    else:
        current_uri = present.get("PrimaryContainer", {}).get("ModelDataUrl")
        current_image = present.get("PrimaryContainer", {}).get("Image")
        assert current_uri == info["uri"] and current_image == IMAGE_URI, (
            "El modelo con ese nombre ya existe, pero apunta a otro artefacto/imagen. No se sobrescribe."
        )
        print("✔ Modelo ya existe, se reutiliza")

    present_cfg = describe_or_none(sm, "describe_endpoint_config", "EndpointConfigName", info["config_name"])
    if present_cfg is None:
        print("Creando EndpointConfig:", info["config_name"])
        sm.create_endpoint_config(
            EndpointConfigName=info["config_name"],
            ProductionVariants=[{
                "VariantName": "AllTraffic",
                "ModelName": info["model_name"],
                "InitialInstanceCount": 1,
                "InstanceType": instance,
                "InitialVariantWeight": 1.0,
            }],
            Tags=[{"Key": "Project", "Value": "bts-flight-delay"}],
        )
        print("✔ Configuración creada")
    else:
        variants = present_cfg["ProductionVariants"]
        assert len(variants) == 1 and variants[0]["ModelName"] == info["model_name"] and variants[0]["InstanceType"] == instance, (
            "Ya existe una configuración para esta versión pero con instancia/modelo diferentes. "
            "No se modifica automáticamente. Usa la misma --instance que al crearla."
        )
        print("✔ Configuración ya existe, se reutiliza")

    current_ep = describe_or_none(sm, "describe_endpoint", "EndpointName", ENDPOINT_NAME)
    if current_ep is None:
        print("Creando Endpoint:", ENDPOINT_NAME)
        sm.create_endpoint(EndpointName=ENDPOINT_NAME, EndpointConfigName=info["config_name"],
                           Tags=[{"Key": "Project", "Value": "bts-flight-delay"}])
        print("✔ Solicitud de creación enviada. Consulta el estado con el comando status.")
    else:
        status = current_ep["EndpointStatus"]
        if current_ep["EndpointConfigName"] != info["config_name"]:
            raise RuntimeError(f"El endpoint {ENDPOINT_NAME} ya usa otra configuración: "
                               f"{current_ep['EndpointConfigName']}. No se cambia automáticamente.")
        print("✔ Endpoint ya existe:", status)
        if status == "Failed":
            print("Causa:", current_ep.get("FailureReason", "Consulta CloudWatch"))
    print("\nEndpoint:", ENDPOINT_NAME)
    print("Modelo:", info["model_name"])
    print("Config:", info["config_name"])
    print("El endpoint se factura mientras está provisionado, aunque cierres SageMaker Notebook.")


def status(session, info, wait_minutes=0):
    sm = session.client("sagemaker")
    deadline = time.monotonic() + wait_minutes * 60
    while True:
        current = describe_or_none(sm, "describe_endpoint", "EndpointName", ENDPOINT_NAME)
        if current is None:
            print("Endpoint no creado aún:", ENDPOINT_NAME)
            return "NotFound"
        status = current["EndpointStatus"]
        print(datetime.now().strftime("%H:%M:%S"), ENDPOINT_NAME, "→", status)
        if status == "Failed":
            print("Causa:", current.get("FailureReason", "Revisa logs de CloudWatch"))
            return status
        if status == "InService":
            print("✔ Puedes probar la API con: python bts_sagemaker_endpoint.py invoke")
            return status
        if not wait_minutes or time.monotonic() >= deadline:
            return status
        time.sleep(25)


def invoke(session, s3, info):
    status_ = status(session, info)
    if status_ != "InService":
        raise RuntimeError("Endpoint no está InService")
    request_key = f"models/flight-delay/{info['version']}/sample_request.json"
    try:
        request = json.loads(s3.get_object(Bucket=BUCKET, Key=request_key)["Body"].read())
    except ClientError as e:
        if _missing(e):
            path = Path("../outputs/model_registry") / info["version"] / "sample_request.json"
            if not path.exists():
                raise RuntimeError(f"No existe solicitud de prueba en s3://{BUCKET}/{request_key}") from e
            request = json.loads(path.read_text())
        else:
            raise
    assert "features" in request and isinstance(request["features"], dict)
    runtime = session.client("sagemaker-runtime")
    start = time.monotonic()
    response = runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="application/json",
        Accept="application/json",
        Body=json.dumps(request, ensure_ascii=False).encode("utf-8"),
    )
    result = json.loads(response["Body"].read().decode("utf-8"))
    latency_ms = round((time.monotonic() - start) * 1000, 2)
    p = result["predictions"][0]["probability_delay_15min"]
    assert isinstance(p, (float, int)) and 0.0 <= p <= 1.0
    assert result["model_version"] == info["version"]
    print("\n✔ Respuesta de la API:")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print("Latencia cliente-medida (incluye red):", latency_ms, "ms")

    # Contrastar la probabilidad del ejemplo contra la primera predicción de test guardada en tuning.
    expected = None
    error = None
    try:
        import numpy as np
        key = TUNING_PREFIX + "deployed_model_test_predictions.npz"
        blob = s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()
        with np.load(io.BytesIO(blob)) as arr:
            expected = float(arr["p"][0])
        error = abs(float(p) - expected)
        if error > 1e-5:
            raise AssertionError(f"Predicción SageMaker no coincide con prueba guardada: {p} vs {expected}")
        print("✔ Coincide con el test guardado en tuning (|diferencia| ≤ 1e-5)")
    except ClientError as e:
        if not _missing(e):
            raise
        print("Aviso: no se encontró NPZ de referencia en S3 para comparar (no impide la invocación)")
    except ImportError:
        print("Aviso: numpy no disponible para comparar con NPZ")

    evidence = {
        "endpoint": ENDPOINT_NAME, "model_version": info["version"],
        "s3_model_uri": info["uri"], "tested_at_utc": datetime.now(timezone.utc).isoformat(),
        "response": result, "latency_client_ms": latency_ms,
        "expected_test_probability": expected, "absolute_probability_difference": error,
        "request_feature_count": len(request["features"]),
    }
    LOCAL_EVIDENCE.mkdir(parents=True, exist_ok=True)
    (LOCAL_EVIDENCE / "endpoint_test_response.json").write_text(
        json.dumps(evidence, indent=2, ensure_ascii=False), encoding="utf-8")
    s3_key = f"modeling/deployment/flight-delay/{info['version']}/endpoint_test_response.json"
    s3.put_object(Bucket=BUCKET, Key=s3_key,
                  Body=json.dumps(evidence, ensure_ascii=False, indent=2).encode(),
                  ContentType="application/json")
    print("✔ Evidencia de consumo guardada en:", f"s3://{BUCKET}/{s3_key}")
    return result


def remove_endpoint(session, info, confirm):
    if not confirm:
        raise ValueError("Para eliminar usa el comando cleanup --confirm (se mantiene S3)")
    sm = session.client("sagemaker")
    current = describe_or_none(sm, "describe_endpoint", "EndpointName", ENDPOINT_NAME)
    if current is not None:
        if current.get("EndpointConfigName") != info["config_name"]:
            raise RuntimeError("No se elimina un endpoint que utiliza una configuración distinta de esta versión")
        print("Eliminando endpoint:", ENDPOINT_NAME)
        sm.delete_endpoint(EndpointName=ENDPOINT_NAME)
        print("Esperando eliminación antes de retirar EndpointConfig y Model...")
        sm.get_waiter("endpoint_deleted").wait(
            EndpointName=ENDPOINT_NAME, WaiterConfig={"Delay":20,"MaxAttempts":60})
    cfg = describe_or_none(sm, "describe_endpoint_config", "EndpointConfigName", info["config_name"])
    if cfg is not None:
        sm.delete_endpoint_config(EndpointConfigName=info["config_name"])
        print("✔ EndpointConfig eliminada")
    model = describe_or_none(sm, "describe_model", "ModelName", info["model_name"])
    if model is not None:
        sm.delete_model(ModelName=info["model_name"])
        print("✔ SageMaker Model eliminado")
    print("✔ Los archivos versionados en S3 se conservan")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["preflight", "deploy", "status", "wait", "invoke", "cleanup"])
    parser.add_argument("--instance", default=DEFAULT_INSTANCE,
                        help="tipo de instancia para endpoint; default ml.t3.medium")
    parser.add_argument("--max-minutes", type=int, default=25)
    parser.add_argument("--confirm", action="store_true", help="necesario solamente para cleanup")
    args = parser.parse_args()
    session = aws()
    s3 = session.client("s3")
    print("Cuenta IAM actual:", session.client("sts").get_caller_identity()["Arn"])
    info = load_info(s3)
    if args.command == "preflight":
        preflight(session, s3, info)
    elif args.command == "deploy":
        preflight(session, s3, info)
        deploy(session, info, args.instance)
    elif args.command == "status":
        status(session, info)
    elif args.command == "wait":
        final = status(session, info, wait_minutes=args.max_minutes)
        if final != "InService":
            print("Estado final:", final, "(puedes volver a ejecutar status sin duplicar recursos)")
    elif args.command == "invoke":
        invoke(session, s3, info)
    elif args.command == "cleanup":
        remove_endpoint(session, info, args.confirm)


if __name__ == "__main__":
    main()
