"""Read explicitly identified two-column spectra, preserving original measurements."""
from pathlib import Path
import numpy as np
import pandas as pd
from lithiumscope.core.hashing import file_sha256
from lithiumscope.core.logging.session import get_logger
from lithiumscope.datasets.adapters.contracts import AdaptedSource
from .wavelengths import wavelength_axis

def read_spectrum(path: Path, *, source: str, sample_id: str, reflectance_unit: str,
                  wavelength_unit: str = "nm", header_lines: int = 0,
                  representation: str = "raw", metadata: dict | None = None) -> AdaptedSource:
    path = Path(path)
    if not sample_id.strip() or representation not in {"raw", "continuum_removed"}:
        raise ValueError("Explicit sample identity and spectral representation required")
    if reflectance_unit not in {"fraction", "percent"} or not isinstance(header_lines, int) or header_lines < 0:
        raise ValueError("Explicit reflectance unit and valid header count required")
    digest = file_sha256(path)
    with path.open(encoding="utf-8-sig") as stream:
        try:
            header = [next(stream).rstrip() for _ in range(header_lines)]
        except StopIteration as exc:
            raise ValueError("Spectrum header is incomplete") from exc
        pairs = []
        for line_number, line in enumerate(stream, header_lines + 1):
            if not line.strip():
                continue
            tokens = line.strip().replace(",", " ").split()
            if len(tokens) != 2:
                raise ValueError(f"Expected two spectral columns at line {line_number}")
            try:
                pairs.append(tuple(map(float, tokens)))
            except ValueError as exc:
                raise ValueError(f"Invalid spectral numbers at line {line_number}") from exc
    if not pairs:
        raise ValueError("Empty spectrum")
    data = np.asarray(pairs)
    wave = wavelength_axis(data[:, 0], unit=wavelength_unit)
    measured = data[:, 1]
    fraction = measured / (100 if reflectance_unit == "percent" else 1)
    if not np.isfinite(fraction).all():
        raise ValueError("Nonfinite reflectance")
    review = (fraction < 0) | (fraction > 1)
    frame = pd.DataFrame({"sample_id": sample_id, "wavelength_nm": wave,
                          "reflectance_raw": measured, "reflectance_fraction": fraction,
                          "physical_review": review})
    if file_sha256(path) != digest:
        raise ValueError("Spectrum changed during reading")
    evidence = {"source": source, "source_file": path.name, "source_sha256": digest,
                "sample_id": sample_id, "representation": representation,
                "reflectance_unit": reflectance_unit, "wavelength_unit": wavelength_unit,
                "header": header, "metadata": metadata or {}, "channels": len(wave),
                "wavelength_min_nm": float(wave[0]), "wavelength_max_nm": float(wave[-1]),
                "out_of_range_channels": int(review.sum()), "training_label_certified": False}
    get_logger("datasets.adapters.spectral").info("Adapted spectrum source=%s sample=%s channels=%d review=%d",
        source, sample_id, len(wave), review.sum())
    return AdaptedSource(frame, evidence)
