"""Descriptive coverage, source dependence and compositional readiness for M1."""
from itertools import combinations

import numpy as np
import pandas as pd

from lithiumscope.core.scientific.eda_statistics import source_series
from lithiumscope.model_1.training.preprocessing.tabular import OXIDES

BINS = [-np.inf, 0, 5, 10, 20, 40, 60, 80, 100, 126.12, 696, np.inf]
_UNKNOWN = {"", "unknown", "none", "nan", "<na>", "not reported", "n/a", "null"}


def _absent(values):
    return values.isna() | values.astype("string").fillna("").str.strip().str.casefold().isin(_UNKNOWN)


def _band(y):
    return pd.cut(y, BINS).astype("string").fillna("missing_or_invalid")


def _correlations(x, y, sources, numeric):
    records = []
    values = x.reindex(columns=numeric).apply(pd.to_numeric, errors="coerce")
    values = values.replace([np.inf, -np.inf], np.nan)
    values["__target__"] = y
    for source, block in values.groupby(sources, sort=True):
        correlation = block.corr(method="spearman", min_periods=3)
        for a, b in combinations(block.columns, 2):
            pairs = int((block[a].notna() & block[b].notna()).sum())
            records.append({"source": source, "left": a, "right": b,
                "pairs": pairs, "spearman": correlation.loc[a,b],
                "scope": "descriptive_not_feature_selection"})
    return pd.DataFrame(records, columns=["source","left","right","pairs","spearman","scope"])


def _categorical_source(x, sources, categorical):
    records, summaries = [], []
    for name in categorical:
        values = x[name].astype("string").mask(_absent(x[name]), "<missing_or_unknown>")
        tab = pd.crosstab(sources, values)
        for source in tab.index:
            for value in tab.columns:
                records.append({"feature": name,"source": source,"value": value,
                    "rows": int(tab.loc[source,value]),
                    "within_source_fraction": float(tab.loc[source,value] / tab.loc[source].sum())})
        array = tab.to_numpy(float)
        n = array.sum()
        dimension = min(array.shape)-1
        score = None
        if n > 0 and dimension > 0:
            expected = np.outer(array.sum(axis=1), array.sum(axis=0))/n
            chi = np.divide((array-expected)**2, expected, out=np.zeros_like(array), where=expected>0).sum()
            score = float(np.sqrt(chi/(n*dimension)))
        summaries.append({"feature": name,"cramers_v_source": score,
            "observed_categories": len(tab.columns),"sources": len(tab.index),
            "interpretation": "source_association_not_causality"})
    return (pd.DataFrame(records,columns=["feature","source","value","rows","within_source_fraction"]),
            pd.DataFrame(summaries,columns=["feature","cramers_v_source","observed_categories","sources","interpretation"]))


def _composition(x, sources):
    ox = x.reindex(columns=OXIDES).apply(pd.to_numeric, errors="coerce")
    finite = np.isfinite(ox)
    complete = finite.all(axis=1)
    positives = complete & ox.gt(0).all(axis=1)
    flags = pd.DataFrame({"complete_ten_oxides": complete,"all_positive": positives,
        "has_zero": (finite & ox.eq(0)).any(axis=1),
        "has_negative": (finite & ox.lt(0)).any(axis=1),
        "incomplete": ~complete}, index=x.index)
    rows = []
    for source, group in flags.groupby(sources, sort=True):
        rows.append({"source": source,"rows": len(group),**{c: int(group[c].sum()) for c in group},
            "ilr_applied": False,"measurement_basis_certified": False})
    return pd.DataFrame(rows)


def coverage_tables(prepared, splits=None):
    x, y = prepared.x, prepared.y
    source = source_series(prepared.frame)
    band = _band(y)
    groups = prepared.groups if prepared.groups is not None else pd.Series("<unknown>",index=x.index)
    publication = prepared.frame.get("source_publication", pd.Series("",index=x.index)).astype("string")
    fallback = prepared.frame.get("Geochemistry Reference",pd.Series("",index=x.index)).astype("string")
    publication = publication.mask(_absent(publication), fallback)
    publication = publication.mask(_absent(publication), "<unknown>")
    cohort = pd.DataFrame({"source": source,"band": band,"spatial_group": groups,"publication": publication})
    source_bands = cohort.groupby(["source","band"],dropna=False).agg(
        rows=("band","size"),groups=("spatial_group","nunique"),
        publication_labels=("publication",lambda v: v[v.ne("<unknown>")].nunique()),
        publication_unknown_rows=("publication",lambda v: int(v.eq("<unknown>").sum()))).reset_index()
    group_bands = cohort.groupby(["source","spatial_group","band"],dropna=False).size().rename("rows").reset_index()
    publication_bands = cohort.groupby(["source","publication","band"],dropna=False).size().rename("rows").reset_index()
    missing = pd.DataFrame({name: _absent(x[name]) for name in x},index=x.index)
    patterns = missing.apply(lambda row: "|".join(str(c) for c in row.index[row.to_numpy(bool)]) or "<none>",axis=1)
    missing_patterns = pd.DataFrame({"source": source,"missing_pattern": patterns,"band":band}).groupby(
        ["source","missing_pattern","band"],dropna=False).size().rename("rows").reset_index()
    frequencies, proxy = _categorical_source(x, source, prepared.schema.categorical)
    tables = {"target_bands_by_source":source_bands,"target_bands_by_group":group_bands,
        "target_bands_by_publication":publication_bands,"missingness_patterns":missing_patterns,
        "spearman_by_source":_correlations(x,y,source,prepared.schema.numeric),
        "categorical_source_frequencies":frequencies,"categorical_source_association":proxy,
        "compositional_readiness":_composition(x,source)}
    if splits is not None:
        records=[]
        for fold,(train,test) in enumerate(splits,1):
            for label,idx in (("TRAIN",train),("TEST_descriptive_only",test)):
                subset=cohort.iloc[idx]
                counts=subset.groupby(["source","band"],dropna=False).agg(
                    rows=("band","size"),groups=("spatial_group","nunique")).reset_index()
                counts.insert(0,"partition",label)
                counts.insert(0,"fold",fold)
                records.append(counts)
        tables["outer_band_coverage"] = pd.concat(records,ignore_index=True)
    return tables
