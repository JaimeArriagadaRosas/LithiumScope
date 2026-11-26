"""Registered M1 research sources, distinct from production dataset provisioning."""
from dataclasses import dataclass


@dataclass(frozen=True)
class ResearchPartition:
    identifier: int
    filename: str


@dataclass(frozen=True)
class ResearchSource:
    key: str
    doi: str
    partitions: tuple[ResearchPartition, ...] = ()
    rock_types: tuple[str, ...] = ()


M1_RESEARCH_SOURCES = (
    ResearchSource("georoc_andean_202609", "10.25625/PVFZCE", (
        ResearchPartition(219128, "2026-09-PVFZCE_ANDEAN_ARC_part1.csv"),
        ResearchPartition(219105, "2026-09-PVFZCE_ANDEAN_ARC_part2.csv"),
        ResearchPartition(219094, "2026-09-PVFZCE_ANDEAN_ARC_part3.csv"),
    )),
    ResearchSource("georoc_felsic_rocks", "10.25625/2JETOA", rock_types=("GRANITE", "RHYOLITE")),
)
