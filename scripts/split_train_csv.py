"""Split a large train.csv into smaller chunks for easier handling."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd


def split_csv(
    source: Path,
    dest_dir: Path,
    rows_per_chunk: int,
    prefix: str = "train_part",
) -> List[Path]:
    if not source.exists():
        raise FileNotFoundError(f"No se encontró el archivo fuente: {source}")

    dest_dir.mkdir(parents=True, exist_ok=True)
    written_files: List[Path] = []

    reader = pd.read_csv(source, chunksize=rows_per_chunk)

    for idx, chunk in enumerate(reader, start=1):
        dest_file = dest_dir / f"{prefix}_{idx:03d}.csv"
        if dest_file.exists():
            raise FileExistsError(
                f"El archivo destino {dest_file} ya existe. Elimina o mueve los "
                "fragmentos anteriores antes de volver a ejecutar el script."
            )
        chunk.to_csv(dest_file, index=False)
        written_files.append(dest_file)
        print(f"[split] Escribió {dest_file} con {len(chunk):,} filas.")

    print(f"[split] Generados {len(written_files)} archivos.")
    return written_files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data/raw/train.csv"),
        help="Ruta al CSV original (por defecto data/raw/train.csv).",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=Path("data/raw"),
        help="Directorio de salida para los fragmentos.",
    )
    parser.add_argument(
        "--rows-per-chunk",
        type=int,
        default=300_000,
        help="Número de filas por archivo resultante (300k por defecto).",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="train_part",
        help="Prefijo base para los archivos generados.",
    )

    args = parser.parse_args()
    split_csv(
        source=args.source,
        dest_dir=args.dest,
        rows_per_chunk=args.rows_per_chunk,
        prefix=args.prefix,
    )


if __name__ == "__main__":
    main()
