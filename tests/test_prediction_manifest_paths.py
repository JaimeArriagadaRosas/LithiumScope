from pathlib import Path

import pandas as pd

from lithiumscope.prediction.integrated import _model_2_cases_from_input
from lithiumscope.prediction.session import PredictionSession


def test_model_2_manifest_resolves_relative_images_from_original_directory(
    monkeypatch,
    tmp_path: Path,
):
    image = tmp_path / "images" / "case.npy"
    image.parent.mkdir()
    image.write_bytes(b"placeholder")

    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(
        [
            {
                "case_id": "case_a",
                "image_path": "images/case.npy",
            }
        ]
    ).to_csv(manifest, index=False)

    session_root = tmp_path / "results"
    session = PredictionSession(
        run_id="test",
        mode="complete",
        root=session_root,
        model_1=session_root / "model_1",
        model_2=session_root / "model_2",
        inputs=session_root / "inputs",
        cross_model=session_root / "cross_model",
        figures=session_root / "cross_model" / "figures",
    )
    for path in (
        session.model_1,
        session.model_2,
        session.inputs,
        session.cross_model,
        session.figures,
    ):
        path.mkdir(parents=True, exist_ok=True)

    cases, info = _model_2_cases_from_input(
        manifest,
        session,
        pd.DataFrame([{"case_id": "case_a"}]),
    )

    assert info["input_type"] == "manifest"
    assert Path(cases.iloc[0]["image_path"]) == image
    assert cases.iloc[0]["sentinel_status"] == "ready"
