import json

import pandas as pd

from lithiumscope.cli.eda_console import show_eda_summary
from lithiumscope.results.eda_artifacts import complete_run


def test_eda_console_reads_verified_profiles_without_modifying_evidence(tmp_path, capsys):
    tables = tmp_path / "tables"
    tables.mkdir()
    pd.DataFrame([
        {"stage": "raw", "source": "GEOROC", "feature": "SiO2", "rows": 4, "missing": 1},
        {"stage": "prepared", "source": "GEOROC", "feature": "SiO2", "rows": 2,
         "missing": 1, "unique_present": 1, "numeric_nonfinite": 0, "min": 3, "median": 3, "max": 3, "std": 0},
    ]).to_csv(tables / "feature_profiles.csv", index=False)
    pd.DataFrame([{"fold": 1, "feature": "SiO2", "train_missing": 1, "train_eligible_rows": 1,
                   "train_unique_present": 0, "train_all_missing": True}]).to_csv(
        tables / "outer_train_features.csv", index=False)
    complete_run(tmp_path, signature_value="test", details={
        "model_group": "model_1", "rows_raw": 4, "rows_prepared": 2, "features": 1})
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    assert show_eda_summary(tmp_path)
    output = capsys.readouterr().out
    assert "SiO2" in output and ">=50% faltantes: 1" in output
    assert "SOLO TRAIN" in output and "NO EVALUADOS" in output
    assert "1. Cohorte canónica" in output and "4. Alcance y evidencias" in output
    assert "Variables preparadas por fuente" not in output
    assert {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()} == before
    (tables / "feature_profiles.csv").write_text("corrupted", encoding="utf-8")
    assert not show_eda_summary(tmp_path)
    assert "Integridad del EDA no verificada" in capsys.readouterr().out
    assert json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))["state"] == "completed"
