"""Separated georoc_links responsibility; preserve prior public API."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lithiumscope.model_2.methodology.physical_label_audit import pd

def georoc_links(man:pd.DataFrame, screen:pd.DataFrame,
                 bibliography:pd.DataFrame,queue:pd.DataFrame)->pd.DataFrame:
    from lithiumscope.model_2.methodology.physical_label_audit import close_numeric, normalized_text, pd
    required={"sample_id","CITATIONS","source_sample","Latitude","Longitude",
              "Li_icpms","raw_rock_class","raw_comment","decimal_lat",
              "decimal_lon","references_complete","screening_tier",
              "geologic_label_certified","source_dois","raw_matches"}
    if not required.issubset(screen):
        raise ValueError(f"GEOROC screen columns missing: {required-set(screen)}")
    if screen.sample_id.duplicated().any() or queue.sample_id.duplicated().any():
        raise ValueError("Ambiguous GEOROC source IDs")
    if not {"ref_id","matched","citation_text","doi"}.issubset(bibliography):
        raise ValueError("Bibliography is missing required metadata")
    lookup=screen.set_index("sample_id")
    queue_ids=set(queue.sample_id.astype(str))
    references=bibliography.set_index(bibliography.ref_id.astype(str))
    rows=[]
    for i,item in man.reset_index(drop=True).iterrows():
        if str(item.sample_id) not in lookup.index:
            raise ValueError(f"GEOROC source-screen match missing: {item.sample_id}")
        src=lookup.loc[str(item.sample_id)]
        coords=close_numeric(item.latitude,src.Latitude) and close_numeric(
            item.longitude,src.Longitude)
        lithium=close_numeric(item.Li_icpms,src.Li_icpms, tol=1e-5)
        citation=str(item.source_citation)==str(src.CITATIONS)
        if not (coords and lithium and citation):
            raise ValueError(f"GEOROC source values disagree for {item.sample_id}")
        if str(item.sample_id) not in queue_ids:
            raise ValueError(f"GEOROC selected queue provenance missing: {item.sample_id}")
        ref_ids=__import__("re").findall(r"\[(\d+)\]",str(src.CITATIONS))
        matched_ref=bool(ref_ids) and all(r in references.index and
                   str(references.loc[r,"matched"]).strip().lower()=="true"
                   for r in ref_ids)
        if not matched_ref:
            raise ValueError(f"GEOROC bibliography match missing: {item.sample_id}")
        if str(src.screening_tier)!="PROVISIONAL_SITE_REVIEW":
            raise ValueError(f"GEOROC screening status not provisional: {item.sample_id}")
        if str(src.geologic_label_certified).strip().lower()=="true":
            raise ValueError("Unexpected certified flag: external field evidence required")
        rock=normalized_text(src.raw_rock_class)
        comment=normalized_text(src.raw_comment)
        decimal_min=int(min(int(src.decimal_lat),int(src.decimal_lon)))
        rows.append({
            "manifest_index":706+int(i),"sample_id":str(item.sample_id),
            "Li_icpms":float(item.Li_icpms),
            "latitude":float(item.latitude),"longitude":float(item.longitude),
            "screening_tier":str(src.screening_tier),
            "source_sample":normalized_text(src.source_sample),
            "citation_ids":str(src.CITATIONS),
            "source_dois":normalized_text(src.source_dois),
            "rock_class_source":rock,
            "raw_location":normalized_text(src.LOCATION),
            "raw_comment":comment,
            "minimum_raw_coordinate_decimals":decimal_min,
            "nominal_coordinate_precision_low":bool(decimal_min<=3),
            "rock_description_missing":bool("NOT GIVEN" in rock.upper() or not rock),
            "raw_comment_missing":not bool(comment),
            "reference_present_in_official_snapshot":bool(matched_ref),
            "source_identity_unique_in_raw":bool(int(src.raw_matches)==1),
            "field_gps_verified":False,
            "exposure_at_patch_verified":False,
            "sample_to_patch_certified":False,
            "source_traceability":"PROVISIONAL_SOURCE_AND_BIBLIOGRAPHY_CROSSCHECKED",
        })
    return pd.DataFrame(rows)
