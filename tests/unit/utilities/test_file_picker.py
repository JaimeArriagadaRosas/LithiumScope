from lithiumscope.cli.file_picker import _windows_filter


def test_windows_filter_converts_space_patterns_to_semicolons():
    value = _windows_filter(
        [
            ("Datos", "*.csv *.xlsx *.xls"),
            ("Todos", "*.*"),
        ]
    )
    assert "*.csv;*.xlsx;*.xls" in value
    assert "Todos (*.*)|*.*" in value
