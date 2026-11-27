"""Fregeneda–Almendra v1 (DOI 10.5281/zenodo.4575375), documented text formats."""
from .reflectance import read_spectrum

def adapt_fregeneda(path, *, instrument, sample_id, representation="raw", metadata=None):
    # Cardoso-Fernandes et al., Data 2021, 6, 33, section 2.
    if representation == "continuum_removed":
        raise ValueError("Continuum-removed export requires a separately verified layout")
    profiles = {"SR-6500": (28, "percent"), "ASD FieldSpec 4": (0, "fraction")}
    if instrument not in profiles:
        raise ValueError("Unsupported or unknown Fregeneda instrument")
    header, unit = profiles[instrument]
    return read_spectrum(path, source="fregeneda_almendra", sample_id=sample_id,
                         header_lines=header, reflectance_unit=unit,
                         representation=representation, metadata={**(metadata or {}), "instrument": instrument})
