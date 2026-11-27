import pytest

from lithiumscope.datasets.management.paths import resolve_image_path


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
