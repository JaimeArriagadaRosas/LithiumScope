from lithiumscope.experiments.analysis.audit_georoc_source_references import (
    parse_bibliography, decimal_places, classify_record,
)

def test_catalogue_parses_id_and_doi():
    data='"[4065] A: PAPER doi: 10.11/x"\n"[24044] B: PAPER doi: 10.22/y"\n'
    refs=parse_bibliography(data)
    assert len(refs)==2
    assert '10.22/y' in refs['24044']

def test_decimal_digits_do_not_certify_gps():
    assert decimal_places('-1.47')==2
    assert decimal_places('-31.4385')==4
    assert decimal_places('-34.1029028192392')==13

def test_surface_sources_do_not_auto_approve_subsurface_samples():
    assert classify_record(True,13,13,1,'DRILL CORE SG246')=='EXCLUDE_SUBSURFACE_LABEL'
    assert classify_record(True,4,4,1,'surface rock')=='PROVISIONAL_SITE_REVIEW'
    assert classify_record(True,4,4,1,'FALCHANI LITHIUM PROJECT')=='MANUAL_SOURCE_REVIEW'
    assert classify_record(True,2,4,1,'')=='MANUAL_SOURCE_REVIEW'
    assert classify_record(False,4,4,1,'')=='MANUAL_SOURCE_REVIEW'
    assert classify_record(True,4,4,60,'')=='MANUAL_SOURCE_REVIEW'
