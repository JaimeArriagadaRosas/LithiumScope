from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def save_real_vs_predicted(y_true,y_pred,destination:Path)->Path:
    destination.parent.mkdir(parents=True,exist_ok=True); y_true=np.asarray(y_true); y_pred=np.asarray(y_pred)
    fig,ax=plt.subplots(figsize=(7,6)); ax.scatter(y_true,y_pred,alpha=0.65)
    low=float(min(y_true.min(),y_pred.min())); high=float(max(y_true.max(),y_pred.max())); ax.plot([low,high],[low,high],linestyle="--")
    ax.set_xlabel("Li_icpms real (ppm)"); ax.set_ylabel("Li_icpms predicho (ppm)"); ax.set_title("LithiumScope — Real vs. predicho"); fig.tight_layout(); fig.savefig(destination,dpi=160); plt.close(fig); return destination
