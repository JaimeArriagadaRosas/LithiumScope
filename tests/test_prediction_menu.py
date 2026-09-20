from pathlib import Path


def test_prediction_menu_has_approved_labels():
    root = Path(__file__).resolve().parents[1]
    text = (
        root
        / "src"
        / "lithiumscope"
        / "cli"
        / "predict_command.py"
    ).read_text(encoding="utf-8")

    assert 'print("1. Modelo 1 — Predicción de concentración")' in text
    assert 'print("2. Modelo 2 — Prospectividad espacial")' in text
    assert 'print("3. Predicción completa")' in text
    assert 'print("4. Demostración integrada automática")' in text
    assert "Modelos 1 + 2" not in text
