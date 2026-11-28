"""Read only explicitly identified, integrity-checked historical M1 studies."""
from pathlib import Path
import json

from lithiumscope.core.hashing import file_sha256

# Identifiers and hashes are frozen in the 26 November scientific closure.
HISTORICAL = {
    "georoc_andean_202609": (
        ("model_1/expansion/20261126T145129_ee7f5431", "0817aac5e5d869e9be0ec9356c8eea189278cf8ec1f896d19e025384affac6b2"),
        ("generalization/model_1_expansion_linked_recovery/20261126T163242_d5f5c5fe", "f8469364dc93133a4fe23edcd385844a426f4d19e94ccf950d6e58365ccd7ce8"),
    ),
    "georoc_felsic_rocks": (
        ("model_1/expansion/20261126T145129_ee7f5431", "0817aac5e5d869e9be0ec9356c8eea189278cf8ec1f896d19e025384affac6b2"),
    ),
    "chen_global_arcs_2020": (
        ("generalization/model_1_chen_profile/20261126T210320_3f80751b", "9d0d0d8e0d59ce63fe6bda01afb2996dd2dfd77c378770a675749abeb029162f"),
        ("generalization/model_1_chen_overlap_audit/20261126T210740_ed33aa31", "b0bfff9a1ee3e7660a5e7388acd2b3ce1886ca6acc1fe6fc7895f057eca44925"),
        ("generalization/model_1_directed_transfer_preflight/20261126T211301_3a7cd8be", "21b112c5faaee0c410a5045ab295b57a41c875eb6dda78f268c8b4c0cfb0d99b"),
        ("generalization/model_1_data_closure/20261126T213014_fc01460c", "929732600e49a8da72ba506729f16a2b7a48195e0fa9189695d2474079762e46"),
    ),
    "meixner_2020": (
        ("generalization/model_1_meixner_identity_audit/20261126T202913_8686ac5d", "39b6218e257284b19061eb1ac9c2b65a65a29e555614f932a57d79c1d6fbd193"),
    ),
    "fernandez_turiel_2020": (
        ("generalization/model_1_directed_whole_rock_audit/20261126T205508_dccd2ea0", "61c1e6c083019ff21ab16cbb54961c864c025d4e571661dd52147f9f933d97ce"),
        ("generalization/model_1_directed_transfer_preflight/20261126T211301_3a7cd8be", "21b112c5faaee0c410a5045ab295b57a41c875eb6dda78f268c8b4c0cfb0d99b"),
    ),
    "macusani_2025": (
        ("generalization/model_1_directed_whole_rock_audit/20261126T205508_dccd2ea0", "61c1e6c083019ff21ab16cbb54961c864c025d4e571661dd52147f9f933d97ce"),
    ),
}


def verify_manifest(folder, expected):
    manifest = folder / "manifest.json"
    if manifest.is_symlink() or file_sha256(manifest) != expected:
        raise ValueError("Historical manifest identity differs")
    record = json.loads(manifest.read_text(encoding="utf-8"))
    outputs = record.get("outputs") or record.get("artifacts")
    if not isinstance(outputs, dict) or not outputs:
        raise ValueError("Historical output inventory unavailable")
    for name, digest in outputs.items():
        path = folder / name
        if path.is_symlink() or not path.resolve().is_relative_to(folder.resolve()) or file_sha256(path) != digest:
            raise ValueError(f"Historical output differs: {name}")
    return record


def inspect_history(key, results_root, base_digest):
    records = []
    for relative, digest in HISTORICAL.get(key, ()):
        folder = Path(results_root) / relative
        result = {"path": relative, "manifest_sha256": digest}
        try:
            manifest = verify_manifest(folder, digest)
            result["state"] = "verified_historical"
            recorded_base = manifest.get("source_sha256")
            result["canonical_source_matches"] = (
                recorded_base == base_digest if isinstance(recorded_base, str) and base_digest else None)
            result["summary"] = manifest.get("summary", {})
        except FileNotFoundError:
            result["state"] = "missing_historical"
        except (OSError, ValueError, TypeError, KeyError) as exc:
            result.update(state="blocked_historical", reason=str(exc))
        records.append(result)
    return records
