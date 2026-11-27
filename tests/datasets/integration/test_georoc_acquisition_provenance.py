from lithiumscope.datasets.georoc_filtered_acquisition import (
    CHEMISTRY,
    _best_chemistry_form,
    _default_payload,
    _parse,
    _select_chemistry,
    acquisition_contract,
)

from lithiumscope.datasets.georoc_query_flow import (
    _is_chem_location_form,
    _is_direct_batch_form,
    compile_file_form,
)

from lithiumscope.tools.georoc.query.georoc_query_models import (
    Form,
    Input,
)

from lithiumscope.tools.georoc.query.georoc_query_actions import (
    resolve_postpage_action,
)

from lithiumscope.tools.georoc.query.georoc_query_diagnostics import (
    diagnose_query_exception,
    format_query_failure,
)


def test_encoding_detection_checks_beyond_first_half_megabyte(
    tmp_path,
):
    from lithiumscope.tools.georoc.processing.georoc_raw_reader import (
        detect_georoc_encoding,
    )

    path = tmp_path / "late_cp1252.csv"
    prefix = (
        "MATERIAL,LI,LONGITUDE,LATITUDE,NOTE\n"
        + "WHOLE ROCK,1,-70,-33,"
        + ("A" * (600 * 1024))
    )
    path.write_bytes(
        (prefix + "\xa0fin\n").encode(
            "cp1252"
        )
    )

    assert detect_georoc_encoding(path) == "cp1252"


def test_source_checkpoint_can_be_reprocessed_without_network(
    tmp_path,
):
    from lithiumscope.tools.georoc.processing.georoc_download_finalize import (
        finish_source_checkpoint,
    )
    from lithiumscope.tools.georoc.query.georoc_query_log import (
        BoundedRunLog,
    )
    from lithiumscope.tools.georoc.processing.georoc_source_checkpoint import (
        source_path_for,
    )

    class SilentSpinner:
        def update(self, message):
            self.message = message

        def succeed(self, message=None):
            self.success = message

    destination = tmp_path / "GEOROC.csv"
    source = source_path_for(destination)
    source.write_bytes(
        (
            "TYPE OF MATERIAL,LI,LONGITUDE,LATITUDE\n"
            "WHOLE ROCK,12,-70.1,-33.4\n"
            "VOLCANIC GLASS,30,-69.9,-22.0\n"
        ).encode("cp1252")
    )
    log = BoundedRunLog(
        tmp_path / "run.log"
    )
    spinner = SilentSpinner()

    path = finish_source_checkpoint(
        destination,
        spinner,
        log,
    )

    assert path == destination
    assert source.is_file()
    frame = __import__("pandas").read_csv(path)
    assert len(frame) == 1
    assert frame.loc[0, "TYPE OF MATERIAL"] == "WHOLE ROCK"
    assert "recuperado" in spinner.success.lower()


def test_schema_profile_reports_material_candidates():
    import pandas as pd
    from lithiumscope.tools.georoc.processing.georoc_schema_profile import (
        profile_georoc_schema,
    )

    frame = pd.DataFrame(
        {
            "TYPE OF MATERIAL": [
                "/ GL",
                "/ GL",
                "WHOLE ROCK",
            ],
            "ROCK NAME": [
                "BASALT",
                "DACITE",
                "ANDESITE",
            ],
            "LI": [10, 20, 30],
        }
    )

    profile = profile_georoc_schema(frame)

    names = {
        candidate.name
        for candidate in profile.candidates
    }
    assert "TYPE OF MATERIAL" in names
    assert "ROCK NAME" in names

    material = next(
        item
        for item in profile.candidates
        if item.name == "TYPE OF MATERIAL"
    )
    assert material.top_values[0] == (
        "/ GL",
        2,
    )


def test_schema_profile_is_logged_before_filter_failure(
    tmp_path,
):
    import pandas as pd
    import pytest
    from lithiumscope.tools.georoc.processing.georoc_postprocess import (
        process_georoc_text_export,
    )
    from lithiumscope.tools.georoc.query.georoc_query_log import (
        BoundedRunLog,
    )
    from lithiumscope.tools.georoc.processing.georoc_schema_profile import (
        log_georoc_schema_profile,
    )

    source = tmp_path / "source.csv"
    pd.DataFrame(
        {
            "TYPE OF MATERIAL": [
                "/ GL [100]",
                "/ GL [101]",
            ],
            "ROCK NAME": [
                "BASALT",
                "DACITE",
            ],
            "LI": [10, 20],
            "LONGITUDE": [-70, -69],
            "LATITUDE": [-33, -22],
        }
    ).to_csv(source, index=False)

    log = BoundedRunLog(
        tmp_path / "run.log"
    )

    with pytest.raises(
        RuntimeError,
        match="WHOLE ROCK",
    ):
        process_georoc_text_export(
            source,
            tmp_path / "out.csv",
            profile_callback=lambda profile: (
                log_georoc_schema_profile(
                    log,
                    profile,
                )
            ),
        )

    text = log.path.read_text(
        encoding="utf-8"
    )
    assert "stage=schema_profile" in text
    assert "TYPE OF MATERIAL" in text
    assert "/ GL [100]" in text


def test_material_parser_understands_georoc_wr_batch_notation():
    from lithiumscope.tools.georoc.processing.georoc_material_parser import (
        parse_georoc_material,
    )

    parsed = parse_georoc_material(
        "/ WR [13695]"
    )

    assert parsed.code == "WR"
    assert parsed.batch_id == "13695"
    assert parsed.known is True


def test_material_parser_understands_georoc_gl_batch_notation():
    from lithiumscope.tools.georoc.processing.georoc_material_parser import (
        parse_georoc_material,
    )

    parsed = parse_georoc_material(
        "/ GL [10206]"
    )

    assert parsed.code == "GL"
    assert parsed.batch_id == "10206"


def test_material_parser_does_not_guess_unknown_values():
    from lithiumscope.tools.georoc.processing.georoc_material_parser import (
        parse_georoc_material,
    )

    parsed = parse_georoc_material(
        "WRONG MATERIAL"
    )

    assert parsed.code is None
    assert parsed.batch_id is None
    assert parsed.known is False


def test_whole_rock_filter_supports_georoc_batch_notation():
    import pandas as pd
    from lithiumscope.tools.georoc.processing.georoc_material_filter import (
        filter_whole_rock,
    )

    frame = pd.DataFrame(
        {
            "MATERIAL": [
                "/ WR [13695]",
                "/ GL [10206]",
                "WHOLE ROCK",
                "VOLCANIC GLASS",
                "",
                "UNEXPECTED",
            ],
            "LI": [1, 2, 3, 4, 5, 6],
        }
    )

    result = filter_whole_rock(frame)

    assert result.rows_before == 6
    assert result.rows_after == 2
    assert result.whole_rock_rows == 2
    assert result.volcanic_glass_rows == 2
    assert result.missing_rows == 1
    assert result.unknown_rows == 1
    assert result.frame["LI"].tolist() == [
        1,
        3,
    ]
