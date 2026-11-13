"""Write M2 v4 scientific conclusion after six algorithms complete."""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd

from lithiumscope.core.paths import PROJECT_ROOT

FOLDERS = {
    "fixed": "comparison_706",
    "nested": "nested_170_706",
    "new_sites": "data_ablation_170_new_sites",
}
ARMS = (
    "random_forest",
    "extra_trees",
    "hist_gradient_boosting",
    "xgboost",
    "catboost",
    "svm_rbf",
)


def document(phase: str, *, root: Path = PROJECT_ROOT) -> Path:
    if phase not in FOLDERS:
        raise ValueError("Unknown M2 experimental phase")
    out = root / "results/scientific_simulations/m2_rededge_v4_20261113" / FOLDERS[phase]
    state = out / ("experiment_manifest.json" if phase == "fixed" else "final_status.json")
    comparison = out / "v4_vs_reconstructed_release_same_706.csv"
    if not state.is_file() or not comparison.is_file():
        raise FileNotFoundError("All six real-image algorithms must finish first")
    s = json.loads(state.read_text(encoding="utf-8"))
    if s.get("status") not in (
        "complete_experimental_development_only",
        "complete_experimental_no_release",
        "completed_exploratory_not_a_release",
    ):
        raise ValueError("Partial experiment cannot be called a scientific result")
    frame = pd.read_csv(comparison).set_index("algorithm")
    if set(frame.index) != set(ARMS) or frame.index.duplicated().any():
        raise ValueError("All six M2 algorithms must be evaluated")
    frame = frame.loc[list(ARMS)]
    for column in ("v4_auc", "v4_ap", "v4_brier", "delta_auc", "delta_ap", "delta_brier"):
        if frame[column].isna().any():
            raise ValueError("Missing M2 v4 metrics")
    top_name = frame.sort_values(["v4_auc", "v4_ap"], ascending=False).index[0]
    top = frame.loc[top_name]
    gains = top.delta_auc > 0 and top.delta_ap > 0 and top.delta_brier <= 0
    table = [
        "| Algoritmo | AUC v4 | AP v4 | Brier v4 | ΔAUC vs release | ΔAP vs release | ΔBrier vs release |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for algorithm, row in frame.iterrows():
        table.append(
            f"| {algorithm} | {row.v4_auc:.5f} | {row.v4_ap:.5f} | "
            f"{row.v4_brier:.5f} | {row.delta_auc:+.5f} | "
            f"{row.delta_ap:+.5f} | {row.delta_brier:+.5f} |"
        )
    judgement = (
        "Mejora descriptiva conjunta para el mayor AUC, aun sin validacion externa"
        if gains
        else "Sin ganancia conjunta de AUC/AP/Brier frente al procedimiento anterior"
    )
    lines = [
        f"# M2 v4: auditoría científica — {phase}",
        "",
        "**No es un release aprobado:** evidencia de desarrollo sin holdout independiente.",
        "",
        "## Diseño y origen",
        "",
        "- 126 características originales intactas + 44 red-edge reales.",
        "- Seis algoritmos: Random Forest, Extra Trees, HGB, XGB, CatBoost, SVM-RBF.",
        "- Mismos 706 TEST históricos; validación por grupos espaciales.",
        "- Baseline: CatBoost de release reentrenado fuera de cada TEST; "
        "no se puntúa el binario publicado sobre su entrenamiento.",
        "- Las etiquetas adicionales GEOROC siguen siendo provisionales.",
        "",
        "## Seis algoritmos frente al release",
        "",
        *table,
        "",
        "## Conclusiones",
        "",
        f"Máximo AUC OOF: **{top_name}**.",
        judgement + ".",
        "Examinar IC95 agrupados por geografía en "
        "§v4_vs_reconstructed_release_same_706.csv§. "
        "La mejora puntual no garantiza superioridad estadística.",
        "La regla oficial por media de folds no equivale a ordenar AUC OOF global.",
        "**Decisión:** mantener el release previo hasta validación externa.",
        "",
        "## Hipótesis registradas",
        "",
        "- H1: bandas red-edge aportan información independiente.",
        "- H2: grupos nuevos corrigen debilidades regionales.",
        "- H3: regularización controla sobreajuste sin sacrificar sensibilidad.",
        "- H4: la cadena M1→Top3→M2→Top1 mejora seleccionar sitios reales.",
        "",
        "## Reproducibilidad",
        "",
        f"- Contrato de origen: §{state.name}§.",
        "- Evidencia: CSV de comparación, predicciones OOF y ternas.",
        "- Plan: §docs/models/M2_EXPANSION_ESPECTRAL_SEIS_ALGORITMOS_20261113.md§.",
    ]
    destination = out / "CONCLUSION_CIENTIFICA.md"
    destination.write_text(
        "\n".join(lines).replace("§", chr(96)) + "\n",
        encoding="utf-8",
    )
    print("M2_V4_CONCLUSION_SAVED", destination, "DEV_GAINS", gains, flush=True)
    return destination


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=FOLDERS, required=True)
    args = parser.parse_args()
    document(args.phase)


if __name__ == "__main__":
    main()
