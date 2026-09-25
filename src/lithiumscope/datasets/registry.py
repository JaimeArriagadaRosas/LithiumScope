from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    model: str
    provider: str
    destination_name: str
    source_url: str
    download_url: str | None = None
    zenodo_record: int | None = None
    zenodo_files: tuple[str, ...] = ()
    large: bool = False
    notes: str = ""


DATASETS: dict[str, DatasetSpec] = {
    "mamani09_public_mirror": DatasetSpec(
        key="mamani09_public_mirror",
        model="model_1",
        provider="direct",
        destination_name="Mamani09_Table_DR2.csv",
        source_url="https://github.com/inshatazeen/Machine-learning-Rocks-Categorisation",
        download_url=(
            "https://raw.githubusercontent.com/inshatazeen/"
            "Machine-learning-Rocks-Categorisation/main/"
            "Mamani09_Table_DR2%20(1).csv"
        ),
        notes=(
            "Public mirror used as a reproducible bootstrap source. It has not been "
            "confirmed as the official OASIS copy used by the original student project."
        ),
    ),
    "georoc_andean_arc": DatasetSpec(
        key="georoc_andean_arc",
        model="model_1",
        provider="manual",
        destination_name="georoc",
        source_url=(
            "https://georoc.eu/georoc/"
            "precompiled/metadata.php?doi=10.25625/PVFZCE"
        ),
        large=True,
        notes=(
            "Optional GEOROC Convergent Margins / Andean Arc source. "
            "The precompiled files are large and are not downloaded "
            "automatically; reviewed CSV exports are placed under "
            "data/raw/model_1/georoc/ before harmonization."
        ),
    ),
    "fregeneda_almendra": DatasetSpec(
        key="fregeneda_almendra",
        model="model_2",
        provider="zenodo",
        destination_name="fregeneda_almendra",
        source_url="https://doi.org/10.5281/zenodo.4575375",
        zenodo_record=4575375,
        notes="Lithium-dedicated spectral library for the Fregeneda-Almendra field.",
    ),
    "greenpeg": DatasetSpec(
        key="greenpeg",
        model="model_2",
        provider="zenodo",
        destination_name="greenpeg",
        source_url="https://doi.org/10.5281/zenodo.6518319",
        zenodo_record=6518319,
        zenodo_files=("0-Database_files.zip", "Metadata.pdf", "Database report.pdf"),
        large=True,
        notes="European pegmatite spectral library; selected archive is large.",
    ),
    "sentinel2": DatasetSpec(
        key="sentinel2",
        model="model_2",
        provider="dynamic",
        destination_name="sentinel2",
        source_url=(
            "https://dataspace.copernicus.eu/data-collections/"
            "copernicus-sentinel-missions/sentinel-2"
        ),
        notes=(
            "Dynamic imagery source. Scenes must be queried from sample coordinates/date/cloud rules; "
            "there is no single static archive to download."
        ),
    ),
}


def get_dataset_spec(key: str) -> DatasetSpec:
    try:
        return DATASETS[key]
    except KeyError as exc:
        raise KeyError(f"Unknown dataset: {key}") from exc


def datasets_for_model(model: str) -> list[DatasetSpec]:
    return [spec for spec in DATASETS.values() if spec.model == model]
