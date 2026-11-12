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


def test_compile_file_form_routes_to_chem_comp_txt():
    html = """
    <form id="FieldItems_comp"
          action="Results.asp"
          method="post">
      <input type="button"
             value="Compile File"
             onclick="postpage('FieldItems_comp','ChemCompTxt.asp');"/>
      <input type="hidden"
             name="Items"
             value="LI + SIO2"/>
      <input type="hidden"
             name="Material"
             value="'WR','GL'"/>
    </form>
    """

    parser = _parse(html)
    form = compile_file_form(
        parser.forms
    )

    assert form is not None
    assert form.action == "ChemCompTxt.asp"
    assert _default_payload(form) == [
        ("Items", "LI + SIO2"),
        ("Material", "'WR','GL'"),
    ]


def test_bounded_run_log_keeps_recent_lines(tmp_path):
    from lithiumscope.tools.georoc.query.georoc_query_log import (
        BoundedRunLog,
    )

    path = tmp_path / "run.log"
    log = BoundedRunLog(
        path,
        max_lines=20,
        max_chars=4000,
    )
    for index in range(40):
        log.event(
            "step",
            f"evento-{index}",
        )

    lines = path.read_text(
        encoding="utf-8"
    ).splitlines()

    assert len(lines) <= 20
    assert any(
        "evento-39" in line
        for line in lines
    )
    assert not any(
        "evento-0" in line
        for line in lines
    )


def test_resolve_download_href_supports_georoc_checkie():
    from lithiumscope.tools.georoc.query.georoc_query_download_links import (
        resolve_download_href,
    )

    href = (
        "javascript:checkIE("
        "'/georoc/results/f08867456202692525710.csv',"
        "'%2Fgeoroc%2Fresults%2Ff08867456202692525710%2Ecsv',"
        "'f08867456202692525710%2Ecsv');"
    )

    assert resolve_download_href(href) == (
        "/georoc/results/f08867456202692525710.csv"
    )


def test_resolve_download_href_leaves_regular_links_unchanged():
    from lithiumscope.tools.georoc.query.georoc_query_download_links import (
        resolve_download_href,
    )

    assert resolve_download_href(
        "/georoc/results/table.csv"
    ) == "/georoc/results/table.csv"


def test_dataset_health_accepts_valid_li_coordinates():
    import pandas as pd
    from lithiumscope.tools.georoc.processing.georoc_dataset_health import (
        assess_dataset_health,
    )

    frame = pd.DataFrame(
        {
            "LI": [12.0, 24.0, None],
            "LONGITUDE": [-70.1, -69.8, -70.0],
            "LATITUDE": [-33.4, -22.1, -91.0],
        }
    )

    health = assess_dataset_health(frame)

    assert health.rows == 3
    assert health.li_nonempty == 2
    assert health.valid_coordinate_pairs == 2
    assert health.complete_li_coordinate_rows == 2
    assert health.out_of_range_coordinates == 1


def test_dataset_health_rejects_dataset_without_usable_li():
    import pandas as pd
    import pytest
    from lithiumscope.tools.georoc.processing.georoc_dataset_health import (
        assess_dataset_health,
    )

    frame = pd.DataFrame(
        {
            "LI": [None, ""],
            "LONGITUDE": [-70.1, -69.8],
            "LATITUDE": [-33.4, -22.1],
        }
    )

    with pytest.raises(
        RuntimeError,
        match="litio utilizables",
    ):
        assess_dataset_health(frame)


def test_raw_reader_falls_back_to_cp1252(tmp_path):
    from lithiumscope.tools.georoc.processing.georoc_raw_reader import (
        read_georoc_raw,
    )

    path = tmp_path / "georoc_cp1252.csv"
    path.write_bytes(
        "MATERIAL,LI,LONGITUDE,LATITUDE,NOTE\n"
        "WHOLE ROCK,12,-70.1,-33.4,Andes\xa0Sur\n".encode(
            "cp1252"
        )
    )

    result = read_georoc_raw(path)

    assert result.encoding == "cp1252"
    assert len(result.frame) == 1


def test_whole_rock_filter_removes_volcanic_glass():
    import pandas as pd
    from lithiumscope.tools.georoc.processing.georoc_material_filter import (
        filter_whole_rock,
    )

    frame = pd.DataFrame(
        {
            "TYPE OF MATERIAL": [
                "WHOLE ROCK",
                "VOLCANIC GLASS",
                "WR",
            ],
            "LI": [10, 20, 30],
        }
    )

    result = filter_whole_rock(frame)

    assert result.rows_before == 3
    assert result.rows_after == 2
    assert result.rows_removed == 1
    assert result.frame["LI"].tolist() == [
        10,
        30,
    ]


def test_postprocess_decodes_filters_and_writes_utf8(
    tmp_path,
):
    import pandas as pd
    from lithiumscope.tools.georoc.processing.georoc_postprocess import (
        process_georoc_text_export,
    )

    raw = tmp_path / "raw.csv"
    raw.write_bytes(
        (
            "TYPE OF MATERIAL,LI,LONGITUDE,LATITUDE,NOTE\n"
            "WHOLE ROCK,12,-70.1,-33.4,Andes\xa0Sur\n"
            "VOLCANIC GLASS,25,-69.9,-22.0,Vidrio\n"
        ).encode("cp1252")
    )
    destination = tmp_path / "filtered.csv"

    result = process_georoc_text_export(
        raw,
        destination,
    )
    frame = pd.read_csv(destination)

    assert result.encoding == "cp1252"
    assert result.rows_before_filter == 2
    assert result.rows_after_filter == 1
    assert frame["TYPE OF MATERIAL"].tolist() == [
        "WHOLE ROCK"
    ]
    assert "Andes Sur" in frame["NOTE"].iloc[0]
