"""Separated run responsibility; preserve prior public API."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lithiumscope.model_2.methodology.satellite_inventory import Path

def run(root:Path,out:Path, *,online:bool=False,
        alternatives:bool=False,workers:int=6)->dict:
    from lithiumscope.model_2.methodology.satellite_inventory import BASE_URL, EXPECTED_SHA, MANIFEST, Path, ScientificScenePolicy, ThreadPoolExecutor, as_completed, fetch_metadata, json, local_inventory, pd, report_alternatives, sha, summarize_scene
    manifest_path=root/MANIFEST
    manifest_hash=sha(manifest_path)
    if manifest_hash!=EXPECTED_SHA:
        raise ValueError("Frozen experimental manifest changed")
    manifest=pd.read_csv(manifest_path)
    if len(manifest)!=783:
        raise ValueError("Expected 783 old/new training labels")
    out.mkdir(parents=True,exist_ok=True)
    local=local_inventory(manifest)
    local.to_csv(out/"local_patch_inventory_783.csv",index=False)
    print("LOCAL_PATCHES_COMPLETE",len(local),
          "unique",local.image_sha256.nunique(),flush=True)
    cache=out/"stac_item_cache"
    scene_ids=sorted(local.scene_id.unique())
    meta=[]
    # Read cached STAC results even in offline mode; remote fetch is opt-in.
    with ThreadPoolExecutor(max_workers=max(1,min(int(workers),8))) as pool:
        futures={pool.submit(fetch_metadata,s,cache,remote=online):s
                 for s in scene_ids}
        for index,future in enumerate(as_completed(futures),1):
            scene_id=futures[future]
            scene,status=future.result()
            if scene:
                record=summarize_scene(scene)
                record["status"]=status
            else:
                record={"scene_id":scene_id,"status":status,
                        "complete_radiometric_metadata":False}
            meta.append(record)
            if index%25==0:print("STAC",index,len(scene_ids),flush=True)
    scenes=pd.DataFrame(meta).sort_values("scene_id")
    scenes.to_csv(out/"scene_metadata_inventory.csv",index=False)
    observed=local.merge(
        scenes[["scene_id","scene_nodata_percent","processing_baseline",
                "complete_radiometric_metadata","status"]],
        on="scene_id",how="left",validate="many_to_one"
    ) if "scene_nodata_percent" in scenes else local.copy()
    if "scene_nodata_percent" in observed:
        observed["scene_nodata_is_high"]=(
            observed.scene_nodata_percent>=30)
    observed.to_csv(out/"training_manifest_scene_audit_783.csv",index=False)
    choices=pd.DataFrame(); audits=pd.DataFrame()
    if alternatives:
        if not online:
            raise ValueError("--alternatives requires --online-metadata")
        choices,audits=report_alternatives(
            manifest,local,output=out,
            policy=ScientificScenePolicy())
    summary={
        "status":"complete",
        "manifest_sha256":manifest_hash,
        "original_images_modified":False,
        "model_or_menu_modified":False,
        "n_samples":len(local),
        "n_unique_images":int(local.image_sha256.nunique()),
        "n_unique_scenes":len(scene_ids),
        "low_coverage_samples":int(local.under_95pct_coverage.sum()),
        "low_coverage_distinct_images":int(
            local.loc[local.under_95pct_coverage,"image_sha256"].nunique()),
        "stac_scenes_retrieved":int(
            (~scenes.status.astype(str).str.startswith(
                ("error","not_requested"))).sum()),
        "stac_scene_metadata_complete":int(scenes.complete_radiometric_metadata.sum()),
        "same_day_inspected_unique_lowquality_patches":int(len(choices)),
        "same_day_accepted_proxy":int(
            (choices.decision=="SAME_DATE_PROXY_ACCEPTED").sum()
            if len(choices) else 0),
        "same_day_abstentions":int(
            (choices.decision=="NO_ACCEPTED_SAME_DAY_SCENE").sum()
            if len(choices) else 0),
        "same_day_remote_errors":int(
            (choices.decision=="REMOTE_SOURCE_ERROR").sum()
            if len(choices) else 0),
        "known_ten_demo_cases_read":False,
        "field_rock_outcrops_certified":0,
        "hypothesis":"Scene coverage and local SCL outrank global scene cloud.",
        "stac_base_url":BASE_URL,
        "selection_rules":{
            "min_sixband_joint_coverage":.95,
            "max_local_SCL_blocked":.25,
            "min_local_SCL_class5_not_vegetated":.5,
            "same_day_only":True,
        },
        "limitations":[
            "Cloud and SCL percentages are scene products, not in-situ verified geology.",
            "Global scene no-data does not equal local patch coverage.",
            "New reflectance features require per-scene STAC raster:bands.",
            "No image replacements or active model changes permitted in this run.",
            "This inventory is not a blind model validation.",
        ],
    }
    (out/"inventory_summary.json").write_text(
        json.dumps(summary,indent=2,ensure_ascii=False),encoding="utf-8")
    print("INVENTORY_COMPLETED",json.dumps({
        k:v for k,v in summary.items() if not isinstance(v,(dict,list))
    }),flush=True)
    return summary
