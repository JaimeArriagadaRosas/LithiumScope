"""M2 local spectral quality and temporal coverage plots."""
from __future__ import annotations

import numpy as np
import pandas as pd

from lithiumscope.results.eda_figures import _figure, save_figures


def render_profiles(directory,frame,target,tables):
    save_figures(directory,frame,target,tables["feature_profiles"],tables["spearman_correlations"])
    if {"sentinel_cloud_cover","sixband_joint_valid_fraction"}<=set(frame.columns):
        cloud=pd.to_numeric(frame.sentinel_cloud_cover,errors="coerce")
        valid=pd.to_numeric(frame.sixband_joint_valid_fraction,errors="coerce")
        keep=np.isfinite(cloud)&np.isfinite(valid)&cloud.between(0,100)&valid.between(0,1)
        if keep.any():
            fig,ax=_figure()
            ax.scatter(cloud[keep],valid[keep],s=12,alpha=.6)
            ax.set(xlabel="Nubosidad de escena (%)",ylabel="Fracción local válida",
                   ylim=(0,1.05),title="Nubosidad de escena no equivale a calidad del parche")
            fig.savefig(directory/"scene_cloud_vs_local_validity.png",dpi=140)
    periods=tables["temporal_profiles"]
    if not periods.empty:
        fig,ax=_figure()
        ax.bar(periods.period_utc,periods.samples)
        ax.tick_params(axis="x",rotation=90)
        ax.set(ylabel="Muestras",title="Distribución temporal de adquisición")
        fig.savefig(directory/"temporal_coverage.png",dpi=140)
