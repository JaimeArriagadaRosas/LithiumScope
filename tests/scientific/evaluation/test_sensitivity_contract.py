"""Research-only eligibility and isolation contracts."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from lithiumscope.experiments.model_1.data_expansion.sensitivity_data import (
    bibliography, candidate_tokens, historical_tokens, reserve, fold_pools)
from lithiumscope.experiments.model_1.studies.expansion.sensitivity import run


def test_requires_explicit_metadata_risk_opt_in():
    with pytest.raises(ValueError,match="allow-unverified-metadata"):
        run("nonexistent")


def test_bibliography_uses_leading_id_not_year_and_requires_exact_partition():
    index=bibliography([{"source":"s","file":"f","section":"references",
                         "cells":["[123] Authors: title [2025] doi: 10.1234/example"]}])
    row=SimpleNamespace(source_dataset="s",source_file="f",source_publication="[123]")
    result=candidate_tokens(row,index)
    assert "georoc:123" in result and "doi:10.1234/example" in result
    assert "georoc:2025" not in result
    row.source_file="different"
    assert not candidate_tokens(row,index)
    row.source_file="f"
    row.source_publication="[123] [999]"
    assert not candidate_tokens(row,index)
    assert "doi:10.1234/example" in historical_tokens(pd.DataFrame(
        {"Geochemistry Reference":["https://doi.org/10.1234/example"]}))[0]


def test_reservation_quarantines_all_linked_publications():
    audit=pd.DataFrame({"sensitivity_eligible":[True]*6,
        "spatial_group":[f"g{i}" for i in range(6)],"screening_state":["high_transfer_review"]*6})
    held,pool=reserve(audit,[{"pub"}]*6,[],[])
    assert held.sum()==1
    assert not pool.any()


def test_fold_pools_block_groups_and_publications_and_keep_high_separate():
    audit=pd.DataFrame({"spatial_group":["test","g1","g2","g3"],
        "screening_state":["convergent_review","convergent_review","high_transfer_review","convergent_review"]})
    x=pd.DataFrame({"SiO2":[70.,71.,72.,73.]})
    pools=fold_pools(audit,x,x,["SiO2"],np.ones(4,dtype=bool),
        [{"a"},{"blocked"},{"c"},{"d"}],{"test"},{"blocked"})
    assert pools["D1"].tolist()==[3]
    assert pools["D2"].tolist()==[2]
    assert pools["D3"].tolist()==[2,3]
    assert pools["D4"].tolist()==[2,3]
