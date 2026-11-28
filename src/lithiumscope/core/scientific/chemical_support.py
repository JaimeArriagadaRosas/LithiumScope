"""TRAIN-only robust chemistry support shared by EDA and experiments."""
import numpy as np
import pandas as pd


def chemical_support(candidate_x, train_x, columns):
    """Robust chemistry-neighbour distance with explicit missingness indicators.

    All location/target variables are excluded from the chemistry index. The
    reference scale and 95th percentile of leave-one-out nearest distances
    are derived only from the supplied reference partition.
    """
    from sklearn.neighbors import NearestNeighbors

    cols=list(columns)
    b=train_x[cols].apply(pd.to_numeric,errors="coerce").replace([np.inf,-np.inf],np.nan)
    c=candidate_x[cols].apply(pd.to_numeric,errors="coerce").replace([np.inf,-np.inf],np.nan)
    median=b.median()
    spread=b.quantile(.75)-b.quantile(.25)
    valid=(median.notna() & spread.gt(0) & b.notna().mean().ge(.5))
    cols=valid.index[valid].tolist()
    if len(cols)<5 or len(b)<2:
        return pd.DataFrame({"chemistry_nn_distance":np.nan,"chemistry_nn_q95":np.nan,
                             "chemistry_nn_supported":False},index=c.index)
    def matrix(frame):
        values=frame[cols]
        scaled=values.sub(median[cols]).div(spread[cols]).fillna(0.)
        return np.concatenate([scaled.to_numpy(float),values.isna().to_numpy(float)],axis=1)
    reference=matrix(b)
    nn=NearestNeighbors(n_neighbors=2).fit(reference)
    baseline=nn.kneighbors(reference,return_distance=True)[0][:,1]
    cutoff=float(np.quantile(baseline,.95))
    distances=nn.kneighbors(matrix(c),n_neighbors=1,return_distance=True)[0][:,0] if len(c) else np.array([])
    observed=c[cols].notna().sum(axis=1)
    supported=(observed.ge(max(5,int(np.ceil(len(cols)*.5)))) & (distances<=cutoff))
    return pd.DataFrame({"chemistry_nn_distance":distances,"chemistry_nn_q95":cutoff,
                         "chemistry_nn_supported":supported},index=c.index)
