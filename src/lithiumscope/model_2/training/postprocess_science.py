"""Separated scientific M2 report; imported lazily for API compatibility."""
from __future__ import annotations

import numpy as np
import pandas as pd


def science_report(root,manifest,folds,y,scores,output):
    from lithiumscope.model_2.training.postprocess import (
        OOD,
        PHYSICAL,
        SCL,
        STAC,
        get_json,
        save_json,
        metrics,
    )
    scl=pd.read_csv(root/SCL)
    ood=pd.read_csv(root/OOD)
    physical=get_json(root/PHYSICAL)
    stac=get_json(root/STAC)
    if len(scl)!=783 or len(ood)!=783:
        raise ValueError("Scientific audit inputs no longer span 783 records")
    if not np.array_equal(manifest.sample_id.astype(str),scl.sample_id.astype(str)):
        raise ValueError("SCL sample ID order differs")
    if not np.array_equal(manifest.sample_id.astype(str),ood.sample_id.astype(str)):
        raise ValueError("OOD sample ID order differs")
    if physical["historical_conflicting_sites"]!=25 or stac["n_unique_scenes"]!=111:
        raise ValueError("Scientific audit identity/status differs")
    sites=(manifest.latitude.round(5).astype(str)+"|"+
           manifest.longitude.round(5).astype(str))
    lab=manifest.Li_icpms.astype(float)>=21.6
    mixed=set(
        pd.DataFrame({"site":sites,"label":lab}).groupby("site")
        .label.nunique().loc[lambda x:x>1].index
    )
    if len(mixed)!=25:
        raise ValueError("Site label conflicts changed")
    qa=pd.DataFrame({
        "sample_id":manifest.sample_id.astype(str),"test_fold":folds,
        "Li_icpms":manifest.Li_icpms.astype(float),
        "fold_specific_high_Li_label":y,
        "source_cohort":np.where(np.arange(783)<706,"historical","GEOROC_provisional"),
        "site":sites,
        "mixed_label_site_at_Li21p6":sites.isin(mixed),
        "image_sha256":scl.image_sha256,
        "six_band_valid_fraction":scl.sixband_joint_valid_fraction,
        "train_quality_pass_95":(scl.sixband_joint_valid_fraction>=.95),
        "SCL_proxy_accepted":scl.scientifically_accepted_proxy.fillna(False).astype(bool),
        "SCL_rejection_reasons":scl.rejection_reasons.fillna(""),
        "SCL_nonvegetated_fraction":scl.scl_class5_not_vegetated_fraction,
        "SCL_blocked_fraction":scl.local_scl_blocked_fraction,
        "OOD_outside_historical_percentile_range":ood.ood_from_706,
        "OOD_above_25pct":ood.ood_from_706>.25,
        "latitude":manifest.latitude,
        "field_geology_verified":False,
    })
    qa.to_csv(output/"scientific_sample_audit_783.csv",index=False)
    masks={
        "all":np.ones(783,bool),
        "historical":qa.source_cohort.eq("historical").to_numpy(),
        "GEOROC_provisional":qa.source_cohort.eq("GEOROC_provisional").to_numpy(),
        "train_qa95_pass":qa.train_quality_pass_95.to_numpy(bool),
        "train_qa95_fail_but_test_retained":~qa.train_quality_pass_95.to_numpy(bool),
        "SCL_accepted_proxy":qa.SCL_proxy_accepted.to_numpy(bool),
        "SCL_abstention":~qa.SCL_proxy_accepted.to_numpy(bool),
        "OOD_over_25pct":qa.OOD_above_25pct.to_numpy(bool),
        "mixed_label_sites":qa.mixed_label_site_at_Li21p6.to_numpy(bool),
        "lat_10S_20S":qa.latitude.between(-20,-10,inclusive="neither").to_numpy(),
        "lat_20S_30S":qa.latitude.between(-30,-20,inclusive="neither").to_numpy(),
    }
    rows=[]
    for name,prob in sorted(scores.items()):
        for subset,mask in masks.items():
            if mask.any():
                rows.append({"algorithm":name,"subset":subset,
                    "validation":"exploratory_OOF_not_field_verified",
                    **metrics(y[mask],prob[mask])})
    pd.DataFrame(rows).to_csv(output/"scientific_stratified_metrics.csv",index=False)
    summary={
        "state":"complete","samples":783,
        "historical":706,"GEOROC_provisional":77,
        "spectral_QA_pass":int(qa.train_quality_pass_95.sum()),
        "spectral_QA_fail_but_test_retained":int((~qa.train_quality_pass_95).sum()),
        "SCL_accepted_as_optical_proxy":int(qa.SCL_proxy_accepted.sum()),
        "SCL_abstentions":int((~qa.SCL_proxy_accepted).sum()),
        "historical_conflicting_label_sites":len(mixed),
        "OOD_above_25pct":int(qa.OOD_above_25pct.sum()),
        "STAC_scenes":stac["n_unique_scenes"],
        "no_field_rock_certifications":True,
        "no_training_labels_changed":True,
        "no_test_cases_removed":True,
        "not_independent_external_validation":True,
    }
    save_json(output/"scientific_diagnostics_summary.json",summary)
    lines=[
        "# Diagnóstico científico de la competencia Modelo 2","",
        "Los hallazgos son DESCRIPTIVOS; no son prueba independiente de campo.",
        "",
        "- Registros: 783 (706 históricos y 77 GEOROC provisionales).",
        "- QA 95%: "+str(summary["spectral_QA_pass"])+
            " válidos para ajuste; "+str(summary["spectral_QA_fail_but_test_retained"])+
            " se retienen en evaluación.",
        "- Proxy óptico SCL: "+str(summary["SCL_accepted_as_optical_proxy"])+
            " aceptados; "+str(summary["SCL_abstentions"])+
            " abstenciones, NO equivalen a roca de campo.",
        "- Sitios históricos con etiquetas contradictorias: 25.",
        "- Fuera de dominio >25% frente al histórico: "+str(summary["OOD_above_25pct"])+".",
        "",
        "Revisar scientific_sample_audit_783.csv y scientific_stratified_metrics.csv.",
        "Las conclusiones no justifican excluir nuevas filas ni promover el modelo.",
    ]
    (output/"SCIENTIFIC_DIAGNOSTICS.md").write_text(
        "\n".join(lines)+"\n",encoding="utf-8")
    return summary
