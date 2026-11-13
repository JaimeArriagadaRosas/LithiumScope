"""Controles de integridad para campaña externa de litio."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

MODULE = Path(__file__).resolve().parents[2] / "scripts" / "external_lithium_evidence_audit.py"
spec = importlib.util.spec_from_file_location("external_lithium_evidence_audit", MODULE)
assert spec is not None and spec.loader is not None
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_invalid_li_values_are_not_accepted():
    assert audit.number("NaN") is None
    assert audit.number("-inf") is None
    assert audit.number("unknown") is None
    assert audit.number("12.5") == 12.5


def test_nearest_site_haversine_distance():
    assert audit.haversine(-70.0, -23.0, -70.0, -23.0) == 0
    assert 100 < audit.haversine(-70.0, -23.0, -70.0, -24.0) < 120


def test_rock_replicates_are_not_counted_as_independent_sites():
    rocks = [
        {"site_key": "A-1", "li_value": 12.0, "longitude": -70.0, "latitude": -23.0},
        {"site_key": "A-1", "li_value": 18.0, "longitude": -70.0, "latitude": -23.0},
        {"site_key": "B-1", "li_value": 25.0, "longitude": -69.0, "latitude": -22.0},
    ]
    sites = audit.site_review(rocks, [(-70.01, -23.01, "train-01")])
    assert len(sites) == 2
    assert sites[0]["replicates"] == 2
    assert sites[0]["li_mean_ppm_AR_ICP_MS"] == 15.0
    assert sites[0]["approved_external_label"] is False
    assert sites[0]["outside_5km"] is False


def test_arcgis_truncation_is_rejected(tmp_path):
    source = tmp_path / "partial.json"
    source.write_text(json.dumps({"features": [
        {"attributes": {"OBJECTID": 1, "LABO_LI": 10.0},
         "geometry": {"x": -70, "y": -24}}
    ]}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="incomplete"):
        audit.arcgis_records(source, 2, "salt_crust", "LABO_LI", 3)


def test_arcgis_error_response_is_rejected(tmp_path):
    source = tmp_path / "error.json"
    source.write_text(json.dumps({"error": {"code": 403}}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="ArcGIS returned an error"):
        audit.arcgis_records(source, 1, "brine", "LABO_LI2", 1)
