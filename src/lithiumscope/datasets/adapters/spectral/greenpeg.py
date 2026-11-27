"""GREENPEG v1 universal-text spectra (DOI 10.5281/zenodo.6518319).

Raw and normalized exports are distinct. The record describes the format
but the local file's precise header and units must be declared by its custodian.
"""
from .reflectance import read_spectrum

def adapt_greenpeg(path, *, sample_id, representation, reflectance_unit, header_lines, metadata=None):
    return read_spectrum(path, source="greenpeg", sample_id=sample_id,
                         representation=representation, reflectance_unit=reflectance_unit,
                         header_lines=header_lines, metadata=metadata)
