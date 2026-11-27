import pytest

from lithiumscope.datasets.management.sources.model_2_paths import resolve_image_path


def test_relocated_path_does_not_fall_back_to_original(tmp_path):
    source = tmp_path / "old"
    root = tmp_path / "new"
    source.mkdir()
    root.mkdir()
    original = source / "a.npy"
    original.write_bytes(b"old")
    local = root / "a.npy"
    local.write_bytes(b"copy")
    registry = {"source_root": str(source), "images": {
        str(original): {"relative_path": "a.npy", "sha256": "test"}}}
    assert resolve_image_path(original, root=root, relocation=registry) == local
    local.unlink()
    with pytest.raises(FileNotFoundError):
        resolve_image_path(original, root=root, relocation=registry)
    with pytest.raises(ValueError, match="not in relocation"):
        resolve_image_path(source / "missing.npy", root=root, relocation=registry)


def test_relocation_rejects_escape(tmp_path):
    root = tmp_path / "new"
    root.mkdir()
    with pytest.raises(ValueError, match="escapes"):
        resolve_image_path("a.npy", root=root, relocation={
            "images": {"a.npy": {"relative_path": "../a.npy"}}})


def test_windows_historical_path_resolves_on_current_platform(tmp_path):
    from lithiumscope.core.hashing import file_sha256
    local = tmp_path / "a.npy"
    local.write_bytes(b"copy")
    original = r"D:\old\patches\a.npy"
    registry = {"source_root": r"D:\old", "images": {
        original: {"relative_path": "a.npy", "sha256": file_sha256(local)}}}
    assert resolve_image_path(r"d:\OLD\patches\a.npy", root=tmp_path, relocation=registry, verify_hash=True) == local
    local.write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        resolve_image_path(original, root=tmp_path, relocation=registry, verify_hash=True)
    with pytest.raises(ValueError, match="not in relocation"):
        resolve_image_path(r"D:\old\missing.npy", root=tmp_path, relocation=registry)
