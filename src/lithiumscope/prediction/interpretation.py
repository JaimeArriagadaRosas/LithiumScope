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


def _fmt_nonnegative(value, digits: int = 3) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "N/D"
    if not math.isfinite(number):
        return "N/D"
    return f"{max(0.0, number):.{digits}f}"


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
            f"{_fmt_nonnegative(interval_low, 2)} a {_fmt(interval_high, 2)} ppm."
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
    try:
        operating_threshold = float(
            row.get("operating_threshold")
        )
    except (TypeError, ValueError):
        operating_threshold = None
    if (
        operating_threshold is not None
        and not math.isfinite(operating_threshold)
    ):
        operating_threshold = None

    text = (
        f"El Modelo 2 asigna un score de prioridad exploratoria de "
        f"{score:.3f} y una categoría descriptiva del score "
        f"{priority}. "
        "La categoría descriptiva usa cortes fijos "
        "(BAJA < 0.40; MEDIA 0.40-0.70; ALTA >= 0.70) y no "
        "equivale a la clasificación operativa positiva/negativa. "
    )
    if operating_threshold is not None:
        operating_label = (
            "POSITIVA"
            if score >= operating_threshold
            else "NEGATIVA"
        )
        text += (
            "Según el umbral operativo del Modelo 2 "
            f"({operating_threshold:.3f}), la clasificación es "
            f"{operating_label}. "
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
            "estima una concentración relativamente alta y el score del Modelo 2 "
            "supera el umbral operativo del Modelo 2. Dentro de los casos evaluados, "
            "este caso merece mayor prioridad relativa para revisión o nuevo muestreo."
        )
    elif status == "concordante_baja":
        text = (
            "Las dos evidencias son concordantes en una prioridad relativa menor: "
            "la estimación geoquímica no supera el umbral experimental y el score "
            "del Modelo 2 tampoco supera su umbral operativo."
        )
    elif status == "divergente_m1_alto":
        text = (
            "Las evidencias divergen: la estimación geoquímica del Modelo 1 es "
            "relativamente alta, pero el score del Modelo 2 no supera su umbral "
            "operativo. La discrepancia es informativa y justifica revisar contexto "
            "geológico, condiciones superficiales y representatividad de la imagen."
        )
    elif status == "divergente_m2_alto":
        text = (
            "Las evidencias divergen: el score del Modelo 2 supera el umbral operativo "
            "del Modelo 2, pero la estimación de concentración del Modelo 1 no supera "
            "el umbral experimental. El sector puede ser interesante para revisión, "
            "pero la señal satelital no debe interpretarse como concentración química."
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


from lithiumscope.prediction.interpretation_summary import build_overall_interpretation
