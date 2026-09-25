from __future__ import annotations

from typing import Iterable


GEOROC_ALIASES: dict[str, tuple[str, ...]] = {
    "source_sample": (
        "SAMPLE NAME", "SAMPLE", "SAMPLE_ID",
        "SAMPLE ID", "UNIQUE_ID", "UNIQUE ID",
    ),
    "Sample_type": (
        "TYPE OF MATERIAL", "SAMPLE TYPE", "MATERIAL",
    ),
    "Rock_type": (
        "ROCK NAME", "ROCK TYPE", "ROCK_NAME",
    ),
    "Longitude": (
        "LONGITUDE", "LONGITUDE (X)", "LON",
    ),
    "Longitude_min": (
        "LONGITUDE MIN", "LONGITUDE MIN.", "LONG MIN",
    ),
    "Longitude_max": (
        "LONGITUDE MAX", "LONGITUDE MAX.", "LONG MAX",
    ),
    "Latitude": (
        "LATITUDE", "LATITUDE (Y)", "LAT",
    ),
    "Latitude_min": (
        "LATITUDE MIN", "LATITUDE MIN.", "LAT MIN",
    ),
    "Latitude_max": (
        "LATITUDE MAX", "LATITUDE MAX.", "LAT MAX",
    ),
    "Age (Ma)": ("AGE(MA)", "AGE (MA)", "AGE"),
    "SiO2": ("SIO2(WT%)", "SIO2 (WT%)", "SIO2"),
    "TiO2": ("TIO2(WT%)", "TIO2 (WT%)", "TIO2"),
    "Al2O3": ("AL2O3(WT%)", "AL2O3 (WT%)", "AL2O3"),
    "Fe2O3": (
        "FE2O3T(WT%)", "FE2O3T (WT%)",
        "FE2O3(WT%)", "FE2O3 (WT%)",
    ),
    "MnO": ("MNO(WT%)", "MNO (WT%)", "MNO"),
    "MgO": ("MGO(WT%)", "MGO (WT%)", "MGO"),
    "CaO": ("CAO(WT%)", "CAO (WT%)", "CAO"),
    "Na2O": ("NA2O(WT%)", "NA2O (WT%)", "NA2O"),
    "K2O": ("K2O(WT%)", "K2O (WT%)", "K2O"),
    "P2O5": ("P2O5(WT%)", "P2O5 (WT%)", "P2O5"),
    "Li_icpms": ("LI(PPM)", "LI (PPM)", "LI_PPM", "LI"),
    "Th_icpms": ("TH(PPM)", "TH (PPM)", "TH"),
    "U_icpms": ("U(PPM)", "U (PPM)", "U"),
    "Rb_icpms": ("RB(PPM)", "RB (PPM)", "RB"),
    "Cs_icpms": ("CS(PPM)", "CS (PPM)", "CS"),
    "Nb_icpms": ("NB(PPM)", "NB (PPM)", "NB"),
    "Ta_icpms": ("TA(PPM)", "TA (PPM)", "TA"),
    "Pb_icpms": ("PB(PPM)", "PB (PPM)", "PB"),
    "Ba_icpms": ("BA(PPM)", "BA (PPM)", "BA"),
    "Sr_icpms": ("SR(PPM)", "SR (PPM)", "SR"),
    "Zr_icpms": ("ZR(PPM)", "ZR (PPM)", "ZR"),
    "V_icpms": ("V(PPM)", "V (PPM)", "V"),
    "Hf_icpms": ("HF(PPM)", "HF (PPM)", "HF"),
}


MAJOR_OXIDES = (
    "SiO2", "TiO2", "Al2O3", "Fe2O3", "MnO",
    "MgO", "CaO", "Na2O", "K2O", "P2O5",
)


def _normalize(value: str) -> str:
    return "".join(
        character
        for character in value.strip().upper()
        if character not in {" ", "_", "-"}
    )


def infer_georoc_column_map(
    columns: Iterable[str],
) -> dict[str, str]:
    actual = {
        _normalize(str(column)): str(column)
        for column in columns
    }
    result: dict[str, str] = {}
    for target, aliases in GEOROC_ALIASES.items():
        for alias in aliases:
            match = actual.get(_normalize(alias))
            if match is not None:
                result[match] = target
                break
    return result
