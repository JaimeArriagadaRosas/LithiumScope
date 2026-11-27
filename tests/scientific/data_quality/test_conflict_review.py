from types import SimpleNamespace
import pandas as pd
import pytest

from lithiumscope.core.hashing import file_sha256
from lithiumscope.model_1.validation.conflict_review import review_register,apply_review


def prepared():
    frame=pd.DataFrame({"source_dataset":["a","a","b"],"Sample":["X","X","Y"],
                        "Li_icpms":[1.,9.,3.],"SiO2":[60.,61.,62.],
                        "Latitude":[-20.,-20.,-21.],"Longitude":[-70.,-70.,-71.]})
    return SimpleNamespace(frame=frame,y=frame.Li_icpms,x=frame[["SiO2"]],
                           target="Li_icpms",schema=SimpleNamespace(numeric=["SiO2"],categorical=[]),
                           audit=SimpleNamespace(dataset_sha256="a"*64))


def test_review_requires_document_and_preserves_all_rows(tmp_path):
    data=prepared()
    ledger=review_register(data)
    assert len(ledger)==2
    mask,audit=apply_review(data,ledger,evidence_root=tmp_path)
    assert mask.all() and audit["unresolved"]==2
    ledger.loc[ledger.index[0],"review_decision"]="exclude_confirmed_incompatible_measurement"
    with pytest.raises(ValueError,match="documentary evidence"):
        apply_review(data,ledger,evidence_root=tmp_path)
    doc=tmp_path/"document.txt"
    doc.write_text("synthetic evidence")
    for col,value in {"reviewer":"test","rationale":"synthetic incompatibility","evidence_locator":"line 1",
                      "evidence_path":doc.name,"evidence_sha256":file_sha256(doc)}.items():
        ledger.loc[ledger.index[0],col]=value
    mask,audit=apply_review(data,ledger,evidence_root=tmp_path)
    assert mask.tolist()==[False,True,True] and len(data.frame)==3
    assert audit["train_exclusions"]==1
    doc.write_text("changed")
    with pytest.raises(ValueError,match="changed"):
        apply_review(data,ledger,evidence_root=tmp_path)
