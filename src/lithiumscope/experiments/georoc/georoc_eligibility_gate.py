"""Evidence gate for GEOROC pilot references: no external test until provenance verified."""
from pathlib import Path
import hashlib
import json
import pandas as pd

BASE=Path("results/scientific_simulations/external_geochemistry_audit")
SOURCES={
 "27313":{"publisher_url":"https://doi.org/10.1016/j.oregeorev.2018.08.034",
          "title":"Lithological and structural controls on the genesis of the Candelaria-Punta del Cobre Iron Oxide Copper Gold district, Northern Chile",
          "bibliography_status":"PUBLISHER_CONFIRMED"},
 "23382":{"publisher_url":"https://doi.org/10.1016/j.gca.2019.09.008",
          "repository_url":"https://ora.ox.ac.uk/objects/uuid:73ede734-fc13-4832-b0cb-a6fbb946706f",
          "title":"New constraints from Central Chile on the origins of enriched continental compositions in thick-crusted arc magmas",
          "bibliography_status":"PUBLISHER_AND_REPOSITORY_CONFIRMED"},
 "29268":{"bibliography_status":"NOT_RESOLVED_IN_AVAILABLE_CITATION_INDEX"},
 "29484":{"bibliography_status":"NOT_RESOLVED_IN_AVAILABLE_CITATION_INDEX"},
}

def run():
    frame=pd.read_csv(BASE/"pilot_target_metadata.csv")
    records=[]
    for row in frame.to_dict("records"):
        ref=str(row["image_id"]).replace("georoc_","")
        evidence=SOURCES[ref]
        records.append({
          "image_id":row["image_id"],"reference_id":ref,
          "sample_name":row["sample_name"],"Li_ppm":row["li_ppm"],
          "nearest_training_km":row["nearest_training_km"],
          "bibliography_status":evidence["bibliography_status"],
          "publisher_url":evidence.get("publisher_url"),
          "repository_url":evidence.get("repository_url"),
          "measurement_method":"NOT_CONFIRMED_FOR_SAMPLE",
          "source_sample_match":"NOT_CONFIRMED_IN_ARTICLE_TABLE",
          "independence_vs_training":"NOT_PROVEN",
          "m2_label_validated":False,
          "approved_for_accuracy_metrics":False,
        })
    report=pd.DataFrame(records)
    report.to_csv(BASE/"pilot_eligibility_gate.csv",index=False)
    detail={"n_pilot":len(report),
            "bibliography_confirmed":int(report.bibliography_status.isin(["PUBLISHER_CONFIRMED","PUBLISHER_AND_REPOSITORY_CONFIRMED"]).sum()),
            "independent_labels_accepted":int(report.approved_for_accuracy_metrics.sum()),
            "citation_index_date":"2024-12","georoc_compilation_date":"2026-09",
            "source_hash":hashlib.sha256((BASE/"pilot_target_metadata.csv").read_bytes()).hexdigest(),
            "limitations":["Methods must be shown for the sample itself, not inferred from the paper title.",
                           "Missing 2026 citations cannot be resolved using the older 2024 citation database.",
                           "Training bootstrap uses Mamani09 mirror; independence requires complete sample/publication overlap check.",
                           "Existing four M2 scores are exploratory stress inference, not independent ROC-AUC/AP/Balanced Accuracy."]}
    (BASE/"pilot_eligibility_gate.json").write_text(json.dumps(detail,indent=2,ensure_ascii=False),encoding="utf8")
    print(json.dumps(detail,indent=2),flush=True)

if __name__=="__main__": run()
