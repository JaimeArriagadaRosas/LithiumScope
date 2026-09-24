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
