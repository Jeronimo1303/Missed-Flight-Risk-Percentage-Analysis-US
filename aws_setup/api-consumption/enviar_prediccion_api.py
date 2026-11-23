#!/usr/bin/env python3
"""Invoca API Gateway (POST /predict) leyendo el payload de un archivo JSON.

Uso:
  1. Crea un archivo .env junto a este script con API_GATEWAY_URL y API_BEARER_TOKEN.
  2. Ejecuta: python enviar_prediccion_api.py sample_request.json

Alternativa:
  python enviar_prediccion_api.py datos.json --url 'https://.../predict' --token '...'

Requiere: pip install requests python-dotenv
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

# Carga .env desde la carpeta del script, independientemente del directorio
# desde el que se invoque; las variables del sistema tienen prioridad.
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env", override=False)


def main():
    parser = argparse.ArgumentParser(description="Envía un JSON a la API BTS de predicción")
    parser.add_argument("archivo", type=Path, help="Archivo JSON con 'features' o 'instances'")
    parser.add_argument("--url", default=os.getenv("API_GATEWAY_URL"), help="URL completa de API Gateway")
    parser.add_argument("--token", default=os.getenv("API_BEARER_TOKEN"), help="Token Bearer (por defecto desde .env)")
    parser.add_argument("--timeout", type=float, default=30, help="Tiempo máximo de espera, segundos")
    parser.add_argument("--output", type=Path, help="Guardar respuesta JSON en archivo")
    args = parser.parse_args()

    if not args.url:
        parser.error("Falta API_GATEWAY_URL o --url")
    if not args.token:
        parser.error("Falta API_BEARER_TOKEN o --token")
    if not args.url.startswith("https://"):
        parser.error("La URL debe comenzar por https://")

    try:
        payload = json.loads(args.archivo.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        parser.error(f"No se puede leer el archivo JSON: {exc}")

    if not isinstance(payload, dict):
        parser.error('El JSON debe ser un objeto con "features" o "instances"')
    if set(payload) == {"features"} and isinstance(payload["features"], dict):
        total = 1
    elif set(payload) == {"instances"} and isinstance(payload["instances"], list):
        total = len(payload["instances"])
        if not (1 <= total <= 100) or not all(isinstance(x, dict) for x in payload["instances"]):
            parser.error("'instances' requiere entre 1 y 100 objetos")
    else:
        parser.error('Formato inválido: usa {"features": {...}} o {"instances": [{...}]}')

    url = args.url.rstrip("/")
    if not url.endswith("/predict"):
        url += "/predict"

    print(f"Enviando {total} vuelo(s) a {url}")
    start = time.perf_counter()
    try:
        response = requests.post(
            url,
            headers={
                "Authorization": f"Bearer {args.token}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            json=payload,
            timeout=args.timeout,
        )
    except requests.RequestException as exc:
        print(f"Error de conexión: {exc}", file=sys.stderr)
        return 1

    elapsed_ms = (time.perf_counter() - start) * 1000
    print(f"HTTP {response.status_code} | {elapsed_ms:.1f} ms")
    try:
        result = response.json()
    except ValueError:
        print(f"Respuesta no JSON: {response.text[:2000]}", file=sys.stderr)
        return 1

    print(json.dumps(result, indent=2, ensure_ascii=False))
    if not response.ok:
        return 1
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Respuesta guardada: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
