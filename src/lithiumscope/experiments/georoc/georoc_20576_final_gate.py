"""Final 20576 source-specific validation gate (no proxy ground-truth inflation)."""
from pathlib import Path
import json,hashlib
import pandas as pd
BASE=Path("results/scientific_simulations/external_geochemistry_audit")
def main():
    cross=pd.read_csv(BASE/"georoc_20576_primary_li_crosswalk.csv")
    all_=pd.read_csv(BASE/"full_mamani_contamination_audit.csv",low_memory=False)
    eligible=cross.loc[cross.same_li & cross.coords_match & cross.passes_filter.eq(1)].copy()
    source=all_.loc[all_.citation_number.eq(20576)].copy()
    merged=eligible.merge(source[["sample_name","nearest_full_mamani_km","drilled_below_surface",
        "plutonic_rock","country","latitude","longitude","ROCK TYPE","LOCATION COMMENT",
        "within_100km_full_mamani","possible_same_coordinate_li"]],on="sample_name",
        validate="one_to_one",how="left")
    if merged.nearest_full_mamani_km.isna().any():raise RuntimeError("Could not match candidate provenance")
    merged["method_specific_li_confirmed"]=False
    merged["surface_lithology_verified"]=False
    merged["independence_confirmed"]=False
    merged["m2_external_accuracy_approved"]=False
    merged["gate_reason"]=merged.apply(lambda r:
        "MAMANI_FULL_BUFFER" if r.within_100km_full_mamani else
        "BURIED" if r.drilled_below_surface else
        "PLUTONIC" if r.plutonic_rock else
        "PRIMARY_METHOD_AND_OUTCROP_UNCONFIRMED",axis=1)
    output=BASE/"georoc_20576_final_candidate_gate.csv"
    merged.to_csv(output,index=False)
    stats={"original_publication_candidates":len(cross),"same_li_source":int(cross.same_li.sum()),
        "same_li_and_coords":int((cross.same_li&cross.coords_match).sum()),
        "original_study_filters_pass":int((cross.same_li&cross.passes_filter.eq(1)).sum()),
        "all_three_preliminary_evidence":len(eligible),
        "full_mamani_100km_pass":int((~merged.within_100km_full_mamani).sum()),
        "approved_for_external_accuracy":0,
        "gate_reason_counts":merged.gate_reason.value_counts().to_dict(),
        "data_sha256":hashlib.sha256(output.read_bytes()).hexdigest(),
        "note":"Independent Li assay, exact geological surface exposure and sample-to-pixel equivalence not demonstrated; external model metrics forbidden."}
    (BASE/"georoc_20576_final_candidate_gate.json").write_text(json.dumps(stats,ensure_ascii=False,indent=2),encoding="utf8")
    print("FINAL_GATE",json.dumps(stats,ensure_ascii=False),flush=True)
    print(merged[["sample_name","li_ppm","nearest_full_mamani_km","gate_reason"]].to_string(index=False),flush=True)
if __name__=="__main__":main()
