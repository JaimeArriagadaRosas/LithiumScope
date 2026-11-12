"""Regression for the accidentally corrupted Spanish CLI messages.

The former menu was restored through a Windows console text transformation
which inserted CP437/CP850 box-drawing bytes into the source UTF-8 strings.
No training is started by these tests.
"""
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from lithiumscope.cli import train_command


def test_training_menu_is_correct_spanish_unicode_without_mojibake(
    monkeypatch,capsys
):
    monkeypatch.setattr(train_command,"choose",lambda *a,**k:"0")
    assert train_command._run_impl() is False
    out=capsys.readouterr().out
    assert "Modelo 1 \u2014 Competencia de predicci\u00f3n de Li" in out
    assert "Modelo 2 \u2014 Competencia de prospectividad espacial" in out
    assert "ÔÇö" not in out and "├│" not in out


def test_training_result_accents_are_preserved(capsys):
    result=SimpleNamespace(
        run_dir=Path("my_run"),
        winner="random_forest",
        failed=[],
        ranking=pd.DataFrame({
            "rank":[1],
            "label":["Random Forest"],
            "status":["ok"],
            "roc_auc_mean":[.624514],
        }),
    )
    train_command._print_outcome(result,"Modelo 2")
    out=capsys.readouterr().out
    assert "Ganador candidato de esta ejecuci\u00f3n: random_forest" in out
    assert "Eval\u00fae el candidato desde M\u00e9tricas y resultados." in out
    assert "ÔÇö" not in out and "├│" not in out


def test_no_encoded_box_drawing_substitutions_in_training_source():
    source=Path(train_command.__file__).read_text(encoding="utf-8")
    for unexpected in ("\u251c","\u2551","\u2502","\xd4\xc7\xf6","\xd4\xe5\xc6"):
        assert unexpected not in source
    assert "RF \u2192 XGBoost \u2192 SVM" in source
