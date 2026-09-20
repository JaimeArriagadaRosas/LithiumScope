import json
from pathlib import Path
import zipfile

from lithiumscope.prediction.release_demo import build_demonstration_release_bundle
import lithiumscope.prediction.release_demo as release_demo


def test_release_bundle_requires_pdf_and_sanitizes_local_paths(monkeypatch, tmp_path: Path):
    project = tmp_path / "project"
    results = project / "results"
    run = results / "predictions" / "demonstration_20260920_140000_m0300"
    run.mkdir(parents=True)

    monkeypatch.setattr(release_demo, "PROJECT_ROOT", project)
    monkeypatch.setattr(release_demo, "RESULTS_DIR", results)

    (run / "report.pdf").write_bytes(b"%PDF-1.4\nplaceholder")
    (run / "prediction.log").write_text(
        f"local path={project}\\data\\file.csv\n",
        encoding="utf-8",
    )
    (run / "prediction_manifest.json").write_text(
        json.dumps(
            {
                "run_id": run.name,
                "mode": "demonstration",
                "created_at_utc": "2026-09-20T17:00:00+00:00",
                "models": {
                    "model_1": {
                        "run_id": "training_a",
                        "algorithm": "catboost",
                        "model_sha256": "a" * 64,
                    },
                    "model_2": {
                        "run_id": "training_b",
                        "algorithm": "random_forest",
                        "model_sha256": "b" * 64,
                    },
                },
                "runtime": {
                    "git_commit": "abc",
                    "hostname": "private-host",
                    "pid": 123,
                },
                "artifacts": {"report_pdf": "report.pdf"},
            }
        ),
        encoding="utf-8",
    )

    archive, checksum = build_demonstration_release_bundle(run.name)

    assert archive.is_file()
    assert checksum.is_file()
    with zipfile.ZipFile(archive) as bundle:
        log_text = bundle.read("prediction.log").decode("utf-8")
        publication = json.loads(bundle.read("release_manifest.json"))
    assert str(project) not in log_text
    assert "<PROJECT_ROOT>" in log_text
    assert "hostname" not in publication["runtime"]
    assert "pid" not in publication["runtime"]
