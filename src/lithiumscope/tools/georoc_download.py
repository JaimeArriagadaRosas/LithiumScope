from __future__ import annotations

import argparse
from pathlib import Path

from lithiumscope.core.paths import DATA_DIR
from lithiumscope.datasets.georoc_acquisition import (
    download_andean_arc,
    discover_andean_arc_files,
)


def _parse_parts(raw: str) -> tuple[int, ...]:
    if raw.strip().lower() == "all":
        return (1, 2, 3)
    values = []
    for item in raw.split(","):
        value = int(item.strip())
        if value not in {1, 2, 3}:
            raise ValueError(
                "Las partes válidas son 1, 2 y 3."
            )
        if value not in values:
            values.append(value)
    if not values:
        raise ValueError(
            "Debe indicar al menos una parte."
        )
    return tuple(values)


def _print_catalog() -> None:
    files = discover_andean_arc_files()
    print("\nGEOROC · ANDEAN ARC")
    total = 0
    for item in files:
        total += item.size_bytes
        print(
            f"Parte {item.part}: {item.filename} | "
            f"{item.size_gib:.2f} GiB | "
            f"{item.persistent_id or 'sin PID'}"
        )
    print(
        f"Total oficial: {total / (1024 ** 3):.2f} GiB"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Descarga de forma reanudable los CSV oficiales "
            "GEOROC Andean Arc usados por LithiumScope."
        )
    )
    parser.add_argument(
        "--parts",
        default="all",
        help=(
            "Partes a descargar: all, 1, 2, 3, "
            "o una lista como 1,2."
        ),
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=(
            DATA_DIR
            / "raw"
            / "model_1"
            / "georoc"
        ),
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Solo muestra los archivos oficiales y sus tamaños.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help=(
            "Verifica MD5 al terminar cuando la API oficial "
            "publique checksum compatible. Esto puede tardar."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Elimina archivos locales/parciales de las partes "
            "seleccionadas y vuelve a descargarlas."
        ),
    )
    args = parser.parse_args()

    _print_catalog()
    if args.list:
        return

    parts = _parse_parts(args.parts)
    destination = args.destination.resolve()

    print(
        "\nDestino: "
        f"{destination}"
    )
    print(
        "La descarga usa archivos .part y se puede reanudar "
        "ejecutando nuevamente el mismo comando."
    )

    results = download_andean_arc(
        destination,
        parts=parts,
        verify_checksum=args.verify,
        force=args.force,
    )

    print("\nDescarga GEOROC completada.")
    for result in results:
        print(
            f"  Parte {result.remote.part}: "
            f"{result.local_path}"
        )
    print(
        "\nSiguiente paso recomendado:\n"
        "  lithiumscope-inspect --step 7"
    )


if __name__ == "__main__":
    main()
