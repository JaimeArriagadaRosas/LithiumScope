from lithiumscope.datasets.georoc_filtered_acquisition import (
    CHEMISTRY,
    _best_chemistry_form,
    _default_payload,
    _parse,
    _select_chemistry,
    acquisition_contract,
)


def test_filtered_georoc_contract_has_no_massive_fallback():
    contract = acquisition_contract()

    assert contract["scope"] == "ANDEAN ARC"
    assert contract["material"] == "WHOLE ROCK"
    assert contract["massive_precompiled_fallback"] is False
    assert "LI" in contract["chemistry"]


def test_chemistry_form_selects_only_requested_analytes():
    html = """
    <html><body>
      <form action="next.asp" method="post">
        <select name="major" multiple>
          <option value="SIO2">SIO2</option>
          <option value="TIO2">TIO2</option>
          <option value="CR2O3">CR2O3</option>
        </select>
        <select name="trace" multiple>
          <option value="LI">LI</option>
          <option value="RB">RB</option>
          <option value="AU">AU</option>
        </select>
        <input type="submit" name="go" value="Continue">
      </form>
    </body></html>
    """

    parser = _parse(html)
    form = _best_chemistry_form(parser.forms)
    payload = _select_chemistry(
        form,
        _default_payload(form),
    )

    selected = set(payload)
    assert ("major", "SIO2") in selected
    assert ("major", "TIO2") in selected
    assert ("trace", "LI") in selected
    assert ("trace", "RB") in selected
    assert ("major", "CR2O3") not in selected
    assert ("trace", "AU") not in selected
    assert set(CHEMISTRY) >= {"LI", "SIO2", "RB"}


from lithiumscope.datasets.georoc_query_flow import (
    _is_chem_location_form,
    _is_direct_batch_form,
    compile_file_form,
)
from lithiumscope.tools.georoc_query_models import (
    Form,
    Input,
)
from lithiumscope.tools.georoc_query_actions import (
    resolve_postpage_action,
)


def test_chemloc_continue_form_does_not_require_submit_name():
    form = Form(
        action="ChemLoc.asp",
        method="post",
        inputs=[
            Input(
                name="",
                value="Continue",
                kind="submit",
                checked=False,
            ),
            Input(
                name="Batches",
                value="46,47,48",
                kind="hidden",
                checked=False,
            ),
            Input(
                name="Matches",
                value="53173",
                kind="hidden",
                checked=False,
            ),
        ],
    )

    assert _is_chem_location_form(form) is True
    assert _default_payload(form) == [
        ("Batches", "46,47,48"),
        ("Matches", "53173"),
    ]





def test_parser_preserves_georoc_postpage_onclick():
    html = """
    <form id="Chemistry"
          action="/georoc/Oceanislands/ChemistrySearch.asp"
          method="post">
      <input type="button"
             value="Convergent Margins"
             onclick="postpage('Chemistry','/georoc/ConvMargin/ChemistrySearch.asp','Convergent Margins');"/>
      <input type="hidden" name="trackcriteria" value=""/>
    </form>
    """

    parser = _parse(html)
    form = parser.forms[0]
    button = form.inputs[0]

    assert "ConvMargin/ChemistrySearch.asp" in button.onclick


def test_resolve_postpage_action_uses_convergent_margin_target():
    html = """
    <form id="Chemistry"
          action="/georoc/Oceanislands/ChemistrySearch.asp"
          method="post">
      <input type="button"
             value="Ocean Islands"
             onclick="postpage('Chemistry','/georoc/Oceanislands/ChemistrySearch.asp','Ocean Islands');"/>
      <input type="button"
             value="Convergent Margins"
             onclick="postpage('Chemistry','/georoc/ConvMargin/ChemistrySearch.asp','Convergent Margins');"/>
      <input type="hidden" name="trackcriteria" value=""/>
    </form>
    """

    parser = _parse(html)
    target, track = resolve_postpage_action(
        parser.forms[0],
        "CONVERGENT MARGINS",
    )

    assert target == "/georoc/ConvMargin/ChemistrySearch.asp"
    assert track == "Convergent Margins"



from lithiumscope.tools.georoc_query_diagnostics import (
    diagnose_query_exception,
    format_query_failure,
)


def test_query_diagnostics_exposes_http_status_url_and_body(
    tmp_path,
    monkeypatch,
):
    import requests

    response = requests.Response()
    response.status_code = 500
    response.url = (
        "https://georoc.eu/georoc/"
        "ConvMargin/ChemistrySearch.asp"
    )
    response._content = b"<html>server error</html>"
    request = requests.Request(
        "POST",
        response.url,
    ).prepare()
    response.request = request

    exc = requests.HTTPError(
        "500 Server Error",
        response=response,
        request=request,
    )

    from lithiumscope.tools import georoc_query_diagnostics as diag

    monkeypatch.setattr(
        diag,
        "LOGS_DIR",
        tmp_path,
    )
    failure = diagnose_query_exception(exc)

    assert failure.status_code == 500
    assert failure.url == response.url
    assert failure.evidence_path is not None
    assert failure.evidence_path.read_text(
        encoding="utf-8"
    ) == "<html>server error</html>"

    formatted = format_query_failure(failure)
    assert "HTTP=500" in formatted
    assert "ConvMargin/ChemistrySearch.asp" in formatted



def test_andean_arc_direct_form_is_detected():
    form = Form(
        action="/georoc/ChemBatchDirect.asp",
        method="post",
        inputs=[
            Input(
                name="BatchesDirect",
                value="34273,58579",
                kind="hidden",
                checked=False,
            ),
            Input(
                name="Items",
                value="LI + SIO2",
                kind="hidden",
                checked=False,
            ),
            Input(
                name="Material",
                value="'WR','GL'",
                kind="hidden",
                checked=False,
            ),
        ],
    )

    assert _is_direct_batch_form(form) is True
    assert _default_payload(form) == [
        ("BatchesDirect", "34273,58579"),
        ("Items", "LI + SIO2"),
        ("Material", "'WR','GL'"),
    ]



def test_resolve_postpage_action_supports_two_arguments():
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
    </form>
    """

    parser = _parse(html)
    target, track = resolve_postpage_action(
        parser.forms[0],
        "COMPILE FILE",
    )

    assert target == "ChemCompTxt.asp"
    assert track == ""



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
    from lithiumscope.tools.georoc_query_log import (
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
    from lithiumscope.tools.georoc_query_download_links import (
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
    from lithiumscope.tools.georoc_query_download_links import (
        resolve_download_href,
    )

    assert resolve_download_href(
        "/georoc/results/table.csv"
    ) == "/georoc/results/table.csv"



def test_dataset_health_accepts_valid_li_coordinates():
    import pandas as pd
    from lithiumscope.tools.georoc_dataset_health import (
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
    from lithiumscope.tools.georoc_dataset_health import (
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
