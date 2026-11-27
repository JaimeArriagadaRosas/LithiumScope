"""M1 source-tail visualization; no model selection from the full cohort."""
from __future__ import annotations

import numpy as np

from lithiumscope.results.eda_figures import _figure, save_figures


def render_profiles(directory, prepared, tables):
    profiles=tables["feature_profiles"]
    save_figures(directory,prepared.frame,prepared.target,
                 profiles[profiles.stage=="prepared"],tables["spearman_correlations"])
    tails=tables["target_tails"]
    data=tails[tails.stage=="prepared"].copy()
    if data.empty:
        return
    valid=np.isfinite(data[["q95","q99","maximum"]]).all(axis=1)
    data=data.loc[valid]
    if data.empty:
        return
    fig,ax=_figure()
    positions=np.arange(len(data))
    for i,key in enumerate(("q95","q99","maximum")):
        ax.bar(positions+(i-1)*.25,data[key],width=.25,label=key)
    ax.set_xticks(positions,data.source,rotation=20,ha="right")
    ax.set(ylabel="Li (ppm)",title="Colas de concentración por fuente (descriptivo)")
    ax.legend()
    fig.savefig(directory/"target_tails_by_source.png",dpi=140)
