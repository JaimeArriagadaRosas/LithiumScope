from lithiumscope.model_1.steps.step_01_load_data import load_data


def test_csv_loader_falls_back_to_cp1252(tmp_path):
    path = tmp_path / "cp1252.csv"
    path.write_bytes("nombre,valor\n¿muestra?,10\n".encode("cp1252"))
    frame = load_data(path)
    assert frame.loc[0, "nombre"] == "¿muestra?"
    assert frame.loc[0, "valor"] == 10
