"""Snapshot integrity-checked descriptive figures alongside prediction outputs."""
from pathlib import Path
import shutil

from lithiumscope.core.hashing import file_sha256
from lithiumscope.core.logging.session import get_logger

logger = get_logger("reporting.figures.eda")


def snapshot_eda_figures(evidence, destination, *, root):
    if evidence.get("state") != "CURRENT_SELECTION_EVIDENCE":
        return []
    sources=[]
    for group,model in evidence["models"].items():
        if model.get("eda"):
            data=model["eda"]
            sources.append(("eda_"+group,root/"results"/group/"eda"/data["run_id"],data))
    for data in evidence.get("research_sources",[]):
        key=data["source"]
        sources.append(("research_"+key,root/"results"/"model_1"/"candidate_eda"/key/data["run_id"],data))
    destination.mkdir(parents=True, exist_ok=True)
    copied=[]
    for prefix,folder,data in sources:
        for rel,digest in sorted(data.get("figures",{}).items()):
            source=folder/rel
            if not source.resolve().is_relative_to(folder.resolve()) or source.is_symlink():
                continue
            if not source.is_file() or file_sha256(source)!=digest:
                logger.warning("EDA figure unavailable or hash mismatch: %s", source)
                continue
            target=destination/(prefix+"_"+Path(rel).name)
            shutil.copyfile(source,target)
            if file_sha256(target)!=digest:
                target.unlink()
                raise ValueError("EDA figure changed during snapshot")
            copied.append(target)
    evidence["figure_snapshot"]={p.name:file_sha256(p) for p in copied}
    logger.info("EDA figures copied: %d", len(copied))
    return copied
