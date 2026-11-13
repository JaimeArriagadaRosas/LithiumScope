r"""Auditoría reproducible de fuentes externas sin modificar el entrenamiento M2.

Uso: python scripts/external_lithium_evidence_audit.py --dataset-root D:\...\2026-11-12
No descarga datos ni autoriza etiquetas para entrenamiento.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


def number(value):
    try:
        n = float(value)
        return n if math.isfinite(n) else None
    except (TypeError, ValueError):
        return None


def haversine(a, b, c, d):
    p1, p2 = math.radians(b), math.radians(d)
    dp, dl = p2 - p1, math.radians(c - a)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 6371.0088 * 2 * math.asin(min(1, math.sqrt(h)))


def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def csv_write(path, rows, fields):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def figshare_records(folder):
    rows = []
    specs = (
        ("samples_rock.csv", "outcrop_rock"),
        ("samples_soil.csv", "surface_soil"),
        ("samples_soil_duplicates.csv", "surface_soil_duplicate"),
    )
    for name, material in specs:
        for source in read_csv(folder / name):
            sample = source.get("sample_id", "").strip()
            site = re.sub(r"R[0-9]+$", "", sample) if material == "outcrop_rock" else sample
            rows.append({
                "source": "FIGSHARE_LA_NEGRA_2026",
                "source_record": sample,
                "site_key": site,
                "material": material,
                "li_value": number(source.get("Li")),
                "li_unit": "ppm",
                "li_flag": source.get("Li_flag", ""),
                "longitude": number(source.get("longitude")),
                "latitude": number(source.get("latitude")),
                "method": "AR-ICP-MS; aqua-regia partial digestion",
                "source_reference": "10.6084/m9.figshare.32181333",
                "admission": "REVIEW_ONLY_NOT_M2_TRAINING",
            })
    return rows


def arcgis_records(path, expected, material, field, layer):
    result = json.loads(path.read_text(encoding="utf-8-sig"))
    if result.get("error"):
        raise RuntimeError(f"ArcGIS returned an error in {path}: {result['error']}")
    features = result.get("features", [])
    if len(features) != expected or result.get("exceededTransferLimit"):
        raise RuntimeError(f"ArcGIS incomplete dataset {path}: {len(features)} vs {expected}")
    rows = []
    for feature in features:
        attrs = feature.get("attributes") or {}
        point = feature.get("geometry") or {}
        rows.append({
            "source": "SERNAGEOMIN_SALARES",
            "source_record": f"{layer}:{attrs.get('OBJECTID')}",
            "site_key": str(attrs.get("COD_MUESTRA") or attrs.get("ID") or attrs.get("OBJECTID")),
            "material": material,
            "li_value": number(attrs.get(field)),
            "li_unit": "NOT_VERIFIED",
            "li_flag": "",
            "longitude": number(point.get("x")),
            "latitude": number(point.get("y")),
            "method": "NOT_VERIFIED",
            "source_reference": attrs.get("AGUA_REF") or attrs.get("SALES_REF") or "",
            "admission": "REVIEW_ONLY_DIFFERENT_MATERIAL",
        })
    return rows


def training_samples(path):
    result = []
    for item in read_csv(path):
        lon = number(item.get("longitude"))
        lat = number(item.get("latitude"))
        if lon is not None and lat is not None and -180 <= lon <= 180 and -90 <= lat <= 90:
            result.append((lon, lat, item.get("sample_id", "")))
    if not result:
        raise ValueError("No reference coordinates could be loaded.")
    return result


def site_review(rock_rows, training):
    grouped = defaultdict(list)
    for row in rock_rows:
        if row["li_value"] is not None:
            grouped[row["site_key"]].append(row)
    out = []
    for site, replicas in sorted(grouped.items()):
        loc = replicas[0]
        x, y = loc["longitude"], loc["latitude"]
        if x is None or y is None:
            continue
        distance, nearest = min(
            (haversine(x, y, tx, ty), tid) for tx, ty, tid in training
        )
        out.append({
            "site_key": site,
            "replicates": len(replicas),
            "li_mean_ppm_AR_ICP_MS": round(
                sum(row["li_value"] for row in replicas) / len(replicas), 6
            ),
            "longitude": x,
            "latitude": y,
            "nearest_training_km": round(distance, 3),
            "nearest_training_sample": nearest,
            "outside_5km": distance >= 5,
            "approved_external_label": False,
        })
    return out


def audit(root, training_manifest):
    if not root.is_dir():
        raise FileNotFoundError(root)
    input_files = (
        root / "fondef-it24i0030-soil-geochem-v1.1.zip",
        root / "figshare_la_negra" / "samples_rock.csv",
        root / "figshare_la_negra" / "samples_soil.csv",
        root / "figshare_la_negra" / "samples_soil_duplicates.csv",
        root / "sernageomin_salmueras.json",
        root / "sernageomin_costras.json",
        root / "sernageomin_limites.json",
    )
    for path in (*input_files, training_manifest):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing or empty evidence: {path}")

    candidates = figshare_records(root / "figshare_la_negra")
    candidates += arcgis_records(
        root / "sernageomin_salmueras.json", 243, "brine", "LABO_LI2", 1
    )
    candidates += arcgis_records(
        root / "sernageomin_costras.json", 369, "salt_crust", "LABO_LI", 3
    )
    boundaries = json.loads((root / "sernageomin_limites.json").read_text(encoding="utf-8"))
    if boundaries.get("error") or len(boundaries.get("features", [])) != 19:
        raise RuntimeError("ArcGIS salar boundary snapshot incomplete or invalid.")

    checks = Counter(item["material"] for item in candidates)
    expected = {"outcrop_rock": 7, "surface_soil": 101,
                "surface_soil_duplicate": 15, "brine": 243, "salt_crust": 369}
    if dict(checks) != expected:
        raise RuntimeError(f"Population mismatch: {dict(checks)} vs {expected}")
    for item in candidates:
        lon, lat = item["longitude"], item["latitude"]
        item["valid_coords"] = (
            lon is not None and lat is not None and
            -180 <= lon <= 180 and -90 <= lat <= 90
        )
        if not item["valid_coords"]:
            item["admission"] = "REJECT_MISSING_OR_INVALID_COORDINATES"
        if item["li_value"] is None or item["li_value"] < 0:
            item["admission"] = "REJECT_INVALID_LI_VALUE"

    rock = [row for row in candidates if row["material"] == "outcrop_rock"]
    sites = site_review(rock, training_samples(training_manifest))
    output = root / "audit"
    output.mkdir(exist_ok=True)
    fields = ["source", "source_record", "site_key", "material", "li_value",
              "li_unit", "li_flag", "longitude", "latitude", "method",
              "source_reference", "valid_coords", "admission"]
    csv_write(output / "external_candidates_separated.csv", candidates, fields)
    csv_write(
        output / "rock_site_review.csv", sites,
        ["site_key", "replicates", "li_mean_ppm_AR_ICP_MS", "longitude",
         "latitude", "nearest_training_km", "nearest_training_sample",
         "outside_5km", "approved_external_label"],
    )
    by_type = {}
    for material in expected:
        sub = [r for r in candidates if r["material"] == material]
        by_type[material] = {
            "rows": len(sub),
            "numeric_li": sum(r["li_value"] is not None and r["li_value"] >= 0 for r in sub),
            "valid_geolocation": sum(r["valid_coords"] for r in sub),
            "unique_coordinates_5dp": len({
                (round(r["longitude"], 5), round(r["latitude"], 5))
                for r in sub if r["valid_coords"]
            }),
        }
    evidence = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "policy": "NO_TRAINING_NO_AUTO_APPROVAL_NO_RELEASE",
        "source_meta": {
            "figshare": {
                "url": "https://doi.org/10.6084/m9.figshare.32181333",
                "license": "CC-BY-4.0", "method": "AR-ICP-MS; partial extraction"
            },
            "sernageomin": {
                "url": "https://services.arcgis.com/r7t1P5pnkoOLRdhr/arcgis/rest/services/Salares_Norte_de_Chile/FeatureServer",
                "license": "Check official reuse conditions before redistribution",
                "note": "Brines and saline crusts are not interchangeable with outcrop rocks"
            },
        },
        "file_sha256": {str(p.relative_to(root)) if p.is_relative_to(root) else str(p): sha256(p)
                        for p in (*input_files, training_manifest)},
        "totals": by_type,
        "salar_polygons": 19,
        "rock_independent_sites_max": len(sites),
        "rock_sites_outside_5km": sum(site["outside_5km"] for site in sites),
        "approved_new_training_labels": 0,
        "training_manifest_rows": len(read_csv(training_manifest)),
        "caveats": [
            "Five rock sites have seven laboratory replicas, not seven independent sites.",
            "Rock assays are aqua-regia-extractable, not necessarily total Li; compatibility requires validation.",
            "SERNAGEOMIN brine and crust Li units/methods are unverified; not pooled with rock labels.",
            "Spatial distance alone does not prove independence or absence of shared publications.",
            "New satellite imagery is not an independent ground-truth label.",
        ],
    }
    (output / "source_integrity.json").write_text(
        json.dumps(evidence, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({
        "output": str(output),
        "source_counts": by_type,
        "rock_sites": len(sites),
        "rock_sites_outside_5km": evidence["rock_sites_outside_5km"],
        "approved_new_training_labels": 0,
    }, indent=2, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--training-manifest", required=True, type=Path)
    args = parser.parse_args()
    audit(args.dataset_root, args.training_manifest)


if __name__ == "__main__":
    main()
