#!/usr/bin/env python3
"""Descarga los prefijos gold/, modeling/ y models/ de un bucket S3.

Estructura final, relativa al directorio donde reside este script:
  data/gold/...         <= s3://BUCKET/gold/...
  outputs/modeling/...  <= s3://BUCKET/modeling/...
  outputs/models/...    <= s3://BUCKET/models/...

Usa credenciales AWS de la cadena estándar de Boto3 (p. ej., LabRole en SageMaker).
Por defecto omite archivos existentes cuyo tamaño coincide con el objeto S3.
Para forzar la descarga, usa --overwrite.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path, PurePosixPath

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError, PartialCredentialsError

DEFAULT_BUCKET = "bts-flights-analysis"
PREFIX_TO_DESTINATION = {
    "gold/": Path("data") / "gold",
    "modeling/": Path("outputs") / "modeling",
    "models/": Path("outputs") / "models",
}
BASE_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Descarga gold/ en data/gold y modeling/ + models/ en outputs/."
    )
    parser.add_argument("--bucket", default=DEFAULT_BUCKET, help="Bucket S3 de origen")
    parser.add_argument("--region", default="us-east-1", help="Región AWS (predeterminado: us-east-1)")
    parser.add_argument("--profile", help="Perfil local de ~/.aws/credentials (opcional)")
    parser.add_argument("--overwrite", action="store_true", help="Sobrescribe incluso archivos con igual tamaño")
    parser.add_argument("--dry-run", action="store_true", help="Muestra qué descargaría sin escribir archivos")
    return parser.parse_args()


def safe_destination(prefix: str, key: str, destination_root: Path) -> Path | None:
    relative = key[len(prefix):]
    if not relative or relative.endswith("/"):
        return None  # Objetos que representan carpetas
    parts = PurePosixPath(relative).parts
    if not parts or any(p in ("..", ".", "") for p in parts):
        raise ValueError(f"Ruta S3 insegura: {key!r}")
    return destination_root.joinpath(*parts)


def main() -> int:
    args = parse_args()
    outputs_dir = BASE_DIR / "outputs"
    if not outputs_dir.is_dir():
        print(f"ERROR: no existe la carpeta requerida: {outputs_dir}", file=sys.stderr)
        print("Créala con 'mkdir outputs' y vuelve a ejecutar el script.", file=sys.stderr)
        return 2

    try:
        session = boto3.Session(profile_name=args.profile, region_name=args.region)
        s3 = session.client(
            "s3",
            config=Config(retries={"mode": "standard", "max_attempts": 5}),
        )
        paginator = s3.get_paginator("list_objects_v2")

        downloaded = skipped = errors = 0
        downloaded_bytes = 0
        for prefix, relative_dest in PREFIX_TO_DESTINATION.items():
            target_root = BASE_DIR / relative_dest
            print(f"\n[S3] s3://{args.bucket}/{prefix}")
            print(f"[LOCAL] {target_root}")
            found = 0

            for page in paginator.paginate(Bucket=args.bucket, Prefix=prefix):
                for obj in page.get("Contents", []):
                    key = obj["Key"]
                    dest = safe_destination(prefix, key, target_root)
                    if dest is None:
                        continue
                    found += 1
                    size = int(obj["Size"])
                    rel_path = dest.relative_to(BASE_DIR)

                    if not args.overwrite and dest.is_file() and dest.stat().st_size == size:
                        skipped += 1
                        print(f"  [OMITIDO] {rel_path}")
                        continue

                    if args.dry_run:
                        print(f"  [SIMULADO] {rel_path} ({size:,} bytes)")
                        continue

                    dest.parent.mkdir(parents=True, exist_ok=True)
                    temp = dest.with_name(dest.name + ".part")
                    try:
                        s3.download_file(args.bucket, key, str(temp))
                        if temp.stat().st_size != size:
                            raise IOError(f"Tamaño incorrecto: {temp.stat().st_size} != {size}")
                        temp.replace(dest)
                        downloaded += 1
                        downloaded_bytes += size
                        print(f"  [DESCARGADO] {rel_path} ({size / (1024**2):.2f} MiB)")
                    except Exception as exc:
                        errors += 1
                        temp.unlink(missing_ok=True)
                        print(f"  [ERROR] {key}: {exc}", file=sys.stderr)
                        if isinstance(exc, (NoCredentialsError, PartialCredentialsError)) or (
                            isinstance(exc, ClientError)
                            and exc.response.get("Error", {}).get("Code") in (
                                "ExpiredToken", "InvalidClientTokenId", "AccessDenied"
                            )
                        ):
                            print("Credenciales vencidas o acceso denegado; renueva el laboratorio y reejecuta.", file=sys.stderr)
                            return 1

            if not found:
                print("  [AVISO] No se encontraron objetos para este prefijo.")

        print("\nResumen:")
        print(f"  Descargados: {downloaded}")
        print(f"  Omitidos por tamaño: {skipped}")
        print(f"  Transferidos: {downloaded_bytes / (1024**2):.2f} MiB")
        print(f"  Errores: {errors}")
        return 1 if errors else 0
    except (ClientError, BotoCoreError, NoCredentialsError, PartialCredentialsError) as exc:
        print(f"ERROR AWS: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
