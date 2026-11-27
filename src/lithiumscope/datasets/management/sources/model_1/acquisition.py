"""Resolve raw GEOROC files; acquire only when explicitly constructing an unfrozen source."""
from lithiumscope.core.logging.session import get_logger


def ensure_georoc_files(pattern, raw_glob, georoc_files):
    logger = get_logger("datasets.management.sources.model_1.acquisition")
    if not georoc_files:
        # Reuse the existing filtered GEOROC acquisition path from the lab.
        # A missing or failed GEOROC source must never silently be skipped.
        from lithiumscope.tools.acquisition.georoc.filtered import (
            GEOROC_FILTERED_NAME,
            acquire_filtered_georoc,
            write_acquisition_contract,
        )

        logger.info("GEOROC missing at %s; acquiring filtered source", raw_glob)
        print("GEOROC no disponible localmente; iniciando descarga filtrada.")
        pattern.parent.mkdir(parents=True, exist_ok=True)
        write_acquisition_contract(
            pattern.parent / "lithiumscope_georoc_query_contract.json"
        )
        acquire_filtered_georoc(
            destination=pattern.parent / GEOROC_FILTERED_NAME
        )
        georoc_files = sorted(pattern.parent.glob(pattern.name))
        if not georoc_files:
            raise RuntimeError(
                "La descarga GEOROC no produjo un CSV utilizable en "
                f"{raw_glob}."
            )
        logger.info("GEOROC available for harmonization: %s", georoc_files)

    return georoc_files
