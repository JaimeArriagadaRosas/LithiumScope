from __future__ import annotations

import math

import pandas as pd


def _fmt(value, digits: int = 3) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "N/D"
    if not math.isfinite(number):
        return "N/D"
    return f"{number:.{digits}f}"


def _corr_strength(value) -> str:
    try:
        magnitude = abs(float(value))
    except (TypeError, ValueError):
        return "no evaluable"
    if not math.isfinite(magnitude):
        return "no evaluable"
    if magnitude < 0.20:
        return "muy débil"
    if magnitude < 0.40:
        return "débil"
    if magnitude < 0.60:
        return "moderada"
    if magnitude < 0.80:
        return "fuerte"
    return "muy fuerte"


def model_1_case_text(row: pd.Series, *, lithium_threshold_ppm: float | None = None) -> str:
    predicted = _fmt(row.get("Li_icpms_predicted"), 2)
    ood = float(row.get("out_of_training_range_fraction", 0.0) or 0.0)
    interval_low = row.get("Li_icpms_interval_low_q90")
    interval_high = row.get("Li_icpms_interval_high_q90")

    text = f"El Modelo 1 estima {predicted} ppm de Li_icpms para esta muestra."
    if pd.notna(interval_low) and pd.notna(interval_high):
        text += (
            f" El intervalo empírico q90 asociado va aproximadamente de "
            f"{_fmt(interval_low, 2)} a {_fmt(interval_high, 2)} ppm."
        )
    if ood > 0.25:
        text += (
            f" La entrada presenta {ood:.0%} de variables numéricas fuera del "
            "dominio empírico observado; la estimación debe interpretarse con cautela."
        )
    else:
        text += " La entrada se mantiene mayoritariamente dentro del dominio empírico observado."

    if lithium_threshold_ppm is not None:
        try:
            high = float(row.get("Li_icpms_predicted")) >= float(lithium_threshold_ppm)
        except (TypeError, ValueError):
            high = False
        if high:
            text += (
                " Respecto del umbral experimental usado por el Modelo 2, la estimación "
                "queda en el grupo de concentración relativamente alta."
            )
        else:
            text += (
                " Respecto del umbral experimental usado por el Modelo 2, la estimación "
                "no queda en el grupo de concentración relativamente alta."
            )
    text += " Esta salida es una estimación predictiva y no sustituye una medición de laboratorio."
    return text


def model_2_case_text(row: pd.Series) -> str:
    score = float(row.get("prospectivity_score", 0.0) or 0.0)
    priority = str(row.get("priority", "desconocida")).upper()
    ood = float(row.get("out_of_training_range_fraction", 0.0) or 0.0)
    text = (
        f"El Modelo 2 asigna un score de prioridad exploratoria de {score:.3f} "
        f"y una categoría {priority}. "
    )
    if priority == "ALTA":
        text += (
            "El entorno presenta una semejanza relativamente alta con los patrones "
            "espaciales y espectrales aprendidos alrededor de referencias enriquecidas."
        )
    elif priority == "MEDIA":
        text += (
            "El entorno presenta evidencia intermedia y conviene interpretarla junto "
            "con antecedentes geoquímicos y geológicos."
        )
    else:
        text += (
            "El entorno presenta una semejanza relativamente baja con los patrones "
            "priorizados por el modelo dentro de su dominio de entrenamiento."
        )
    if ood > 0.25:
        text += (
            f" Además, {ood:.0%} de las características están fuera del rango empírico "
            "observado, por lo que la prioridad tiene menor respaldo de aplicabilidad."
        )
    text += (
        " El score es una prioridad relativa: no representa concentración de litio "
        "ni probabilidad de existencia de un yacimiento."
    )
    return text


def integrated_case_text(row: pd.Series) -> str:
    status = str(row.get("concordance", ""))
    ood_m1 = float(row.get("model_1_out_of_training_range_fraction", 0.0) or 0.0)
    ood_m2 = float(row.get("model_2_out_of_training_range_fraction", 0.0) or 0.0)

    if status == "concordante_alta":
        text = (
            "Las dos evidencias apuntan en una dirección compatible: el Modelo 1 "
            "estima una concentración relativamente alta y el Modelo 2 asigna una "
            "señal espacial/espectral elevada. Dentro de los casos evaluados, este "
            "caso merece mayor prioridad relativa para revisión o nuevo muestreo."
        )
    elif status == "concordante_baja":
        text = (
            "Las dos evidencias son concordantes en una prioridad relativa menor: "
            "la estimación geoquímica no supera el umbral experimental y la señal "
            "espacial/espectral tampoco lo supera."
        )
    elif status == "divergente_m1_alto":
        text = (
            "Las evidencias divergen: la estimación geoquímica del Modelo 1 es "
            "relativamente alta, pero el Modelo 2 no asigna una señal espacial/espectral "
            "equivalente. La discrepancia es informativa y justifica revisar contexto "
            "geológico, condiciones superficiales y representatividad de la imagen."
        )
    elif status == "divergente_m2_alto":
        text = (
            "Las evidencias divergen: el Modelo 2 asigna una señal espacial/espectral "
            "elevada, pero la estimación de concentración del Modelo 1 no supera el "
            "umbral experimental. El sector puede ser interesante para revisión, pero "
            "la señal satelital no debe interpretarse como concentración química."
        )
    else:
        text = (
            "No existe información emparejada suficiente para formular una lectura "
            "integrada de este caso."
        )

    if ood_m1 > 0.25 or ood_m2 > 0.25:
        text += (
            " Al menos uno de los modelos está operando parcialmente fuera de su dominio "
            "empírico, por lo que la conclusión integrada debe tratarse con cautela."
        )
    return text


def build_overall_interpretation(
    *,
    model_1_metrics: dict,
    model_2_metrics: dict,
    correlations: pd.DataFrame,
    concordance: pd.DataFrame,
    overlap_audit: dict | None = None,
    model_1_case_count: int | None = None,
    model_2_case_count: int | None = None,
    paired_case_count: int | None = None,
) -> str:
    lines = [
        "INTERPRETACIÓN CIENTÍFICA INTEGRADA",
        "",
        "Pregunta del Modelo 1: ¿qué concentración de Li_icpms podría presentar una muestra compatible?",
        "Pregunta del Modelo 2: ¿qué sectores muestran patrones espaciales y espectrales semejantes a "
        "los observados alrededor de muestras relativamente enriquecidas?",
        "",
    ]

    if model_1_case_count is not None or model_2_case_count is not None:
        lines.extend(
            [
                "Cobertura de la ejecución:",
                f"- Casos evaluados por Modelo 1: {model_1_case_count if model_1_case_count is not None else 'N/D'}",
                f"- Casos evaluados por Modelo 2: {model_2_case_count if model_2_case_count is not None else 'N/D'}",
                f"- Casos emparejados: {paired_case_count if paired_case_count is not None else 'N/D'}",
                "",
            ]
        )

    evaluated_counts = [
        value
        for value in (model_1_case_count, model_2_case_count)
        if value is not None
    ]
    if evaluated_counts and min(evaluated_counts) < 30:
        lines.extend(
            [
                "Advertencia de tamaño muestral:",
                "- Esta evaluación contiene pocos casos y debe interpretarse como demostración externa exploratoria, no como validación definitiva.",
                "",
            ]
        )

    if model_1_metrics:
        lines.append(
            "Modelo 1 — evaluación con verdad de referencia: "
            f"RMSE={_fmt(model_1_metrics.get('rmse'))} ppm, "
            f"MAE={_fmt(model_1_metrics.get('mae'))} ppm, "
            f"R²={_fmt(model_1_metrics.get('r2'))}."
        )
    else:
        lines.append(
            "Modelo 1 — no se proporcionó verdad de referencia suficiente para calcular métricas externas."
        )

    if model_2_metrics:
        lines.append(
            "Modelo 2 — evaluación con etiquetas derivadas del umbral experimental: "
            f"ROC-AUC={_fmt(model_2_metrics.get('roc_auc'))}, "
            f"Average Precision={_fmt(model_2_metrics.get('average_precision'))}, "
            f"Balanced Accuracy={_fmt(model_2_metrics.get('balanced_accuracy'))}."
        )
    else:
        lines.append(
            "Modelo 2 — no se dispuso de verdad de referencia suficiente para calcular métricas externas."
        )

    if not concordance.empty and "concordance" in concordance.columns:
        counts = concordance["concordance"].value_counts().to_dict()
        lines.extend(
            [
                "",
                "Lectura conjunta de casos emparejados:",
                f"- Concordancia alta: {counts.get('concordante_alta', 0)}",
                f"- Concordancia baja: {counts.get('concordante_baja', 0)}",
                f"- Divergencia M1 alto / M2 bajo: {counts.get('divergente_m1_alto', 0)}",
                f"- Divergencia M2 alto / M1 bajo: {counts.get('divergente_m2_alto', 0)}",
            ]
        )
        top = concordance[concordance["concordance"] == "concordante_alta"].copy()
        if not top.empty and "prospectivity_score" in top.columns:
            top = top.sort_values(
                ["prospectivity_score", "Li_icpms_predicted"],
                ascending=[False, False],
            )
            names = ", ".join(top["case_id"].astype(str).head(5))
            lines.append(
                "Casos con mayor prioridad relativa por evidencia convergente: "
                + names
                + "."
            )

    if not correlations.empty:
        valid = correlations.dropna(subset=["pearson"])
        if not valid.empty:
            strongest = valid.iloc[valid["pearson"].abs().argmax()]
            lines.extend(
                [
                    "",
                    "Relación cuantitativa destacada:",
                    f"- {strongest['relationship']}: Pearson={_fmt(strongest['pearson'])} "
                    f"({_corr_strength(strongest['pearson'])}), "
                    f"Spearman={_fmt(strongest['spearman'])}, n={int(strongest['n'])}.",
                    "Las correlaciones describen asociación estadística y no causalidad.",
                ]
            )

    if overlap_audit:
        lines.extend(
            [
                "",
                "Auditoría de independencia del conjunto de demostración:",
                f"- Estado: {overlap_audit.get('status', 'desconocido')}",
                f"- Coincidencias exactas de identificador: {overlap_audit.get('sample_id_matches', 'N/D')}",
                f"- Coincidencias aproximadas por coordenadas: {overlap_audit.get('coordinate_matches', 'N/D')}",
            ]
        )

    lines.extend(
        [
            "",
            "Conclusión de alcance:",
            "LithiumScope combina dos respuestas independientes para apoyar priorización preliminar. "
            "La convergencia aumenta el interés relativo dentro del conjunto analizado, pero no confirma "
            "un recurso, no sustituye ICP-MS y no reemplaza evaluación geológica, mineralógica ni de terreno.",
        ]
    )
    return "\n".join(lines)
