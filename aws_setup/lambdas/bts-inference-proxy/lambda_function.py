"""Puente protegido: API Gateway HTTP API -> AWS Lambda (LabRole) -> SageMaker.

Ruta: POST /predict, payloadFormatVersion 2.0.
El endpoint SageMaker recibe características YA calculadas para T-240 minutos.
La API requiere Authorization: Bearer <API_SHARED_TOKEN>.
"""
import base64
import hmac
import json
import logging
import os
from time import monotonic

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError

LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)
REGION = os.getenv("AWS_REGION", "us-east-1")
ENDPOINT = os.getenv("SAGEMAKER_ENDPOINT", "bts-flight-delay-api")
MAX_BODY_BYTES = 128 * 1024
MAX_BATCH = 100


def respond(status, obj):
    return {
        "statusCode": status,
        "headers": {
            "content-type": "application/json; charset=utf-8",
            "cache-control": "no-store",
        },
        "isBase64Encoded": False,
        "body": json.dumps(obj, ensure_ascii=False, allow_nan=False),
    }


def parse_payload(event):
    raw = event.get("body")
    if raw is None:
        raise ValueError("Se requiere un cuerpo JSON")
    if event.get("isBase64Encoded", False):
        raw_bytes = base64.b64decode(raw, validate=True)
    elif isinstance(raw, str):
        raw_bytes = raw.encode("utf-8")
    else:
        raise ValueError("Cuerpo inválido")
    if not raw_bytes or len(raw_bytes) > MAX_BODY_BYTES:
        raise ValueError("El cuerpo debe tener entre 1 y 131072 bytes")
    payload = json.loads(raw_bytes.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Se requiere un objeto JSON")

    if set(payload.keys()) == {"features"} and isinstance(payload["features"], dict):
        rows = [payload["features"]]
    elif set(payload.keys()) == {"instances"} and isinstance(payload["instances"], list):
        rows = payload["instances"]
    else:
        raise ValueError('El formato debe ser {"features": {...}} o {"instances": [...]}')
    if not 1 <= len(rows) <= MAX_BATCH or any(not isinstance(r, dict) for r in rows):
        raise ValueError("El lote debe contener entre 1 y 100 objetos JSON")

    # Reenvía exactamente el formato aceptado por inference.py, sin alterar valores.
    return payload, len(rows)


def lambda_handler(event, context):
    # Si no hay secreto configurado, no deja la función sin protección.
    expected_token = os.getenv("API_SHARED_TOKEN", "")
    if len(expected_token) < 32:
        LOG.error("Falta configurar API_SHARED_TOKEN (mínimo 32 caracteres)")
        return respond(503, {"error": "API no configurada"})

    headers = {str(k).lower(): str(v) for k, v in (event.get("headers") or {}).items()}
    authorization = headers.get("authorization", "")
    provided = authorization[7:] if authorization.startswith("Bearer ") else ""
    if not hmac.compare_digest(provided, expected_token):
        return respond(401, {"error": "No autorizado"})

    method = (event.get("requestContext") or {}).get("http", {}).get("method")
    path = event.get("rawPath") or "/predict"
    if method != "POST" or path != "/predict":
        return respond(404, {"error": "Ruta no encontrada"})
    if headers.get("content-type", "").split(";")[0].strip().lower() != "application/json":
        return respond(415, {"error": "Usa Content-Type: application/json"})

    try:
        payload, number = parse_payload(event)
        body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (ValueError, TypeError, UnicodeError, json.JSONDecodeError) as exc:
        return respond(400, {"error": "Solicitud inválida", "detail": str(exc)[:220]})

    started = monotonic()
    try:
        runtime = boto3.client(
            "sagemaker-runtime",
            region_name=REGION,
            config=Config(
                connect_timeout=3,
                read_timeout=18,
                retries={"total_max_attempts": 1},
            ),
        )
        result = runtime.invoke_endpoint(
            EndpointName=ENDPOINT,
            ContentType="application/json",
            Accept="application/json",
            Body=body,
        )
        content = result["Body"].read().decode("utf-8")
        decoded = json.loads(content)
        if not isinstance(decoded, dict) or not isinstance(decoded.get("predictions"), list):
            raise ValueError("Respuesta del endpoint inesperada")
        if len(decoded["predictions"]) != number:
            raise ValueError("Número de resultados inesperado")
        LOG.info("Inferencia exitosa: registros=%d duracion_ms=%.1f", number, (monotonic()-started)*1000)
        return respond(200, decoded)
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "Unknown")
        # No se expone el error interno ni las variables enviadas al cliente.
        LOG.exception("Fallo SageMaker [%s]", code)
        if code == "ModelError":
            return respond(422, {"error": "El modelo rechazó las variables; revisa el contrato"})
        return respond(502, {"error": "SageMaker no está disponible para esta solicitud"})
    except (BotoCoreError, TimeoutError) as exc:
        LOG.exception("Fallo de comunicación con SageMaker")
        return respond(504, {"error": "Tiempo de espera del modelo agotado"})
    except (ValueError, UnicodeError, KeyError, json.JSONDecodeError):
        LOG.exception("Respuesta inválida del endpoint")
        return respond(502, {"error": "Respuesta inválida del modelo"})
