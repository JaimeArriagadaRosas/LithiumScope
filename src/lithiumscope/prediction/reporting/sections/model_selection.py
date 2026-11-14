"""Scientific selection justification in the ORIGINAL integrated PDF builder."""
from __future__ import annotations

from reportlab.lib.units import cm
from reportlab.platypus import PageBreak, Spacer

from lithiumscope.prediction.reporting.primitives import paragraph, table


def _number(x, digits=3):
    if x is None:
        return "N/D"
    try:
        return f"{float(x):.{digits}f}".replace(".", ",")
    except (ValueError, TypeError):
        return "N/D"


def _pct(x):
    return _number(float(x) * 100, 2) + " %" if x is not None else "N/D"


def _run_label(run_id, group):
    day = run_id.split("_")[1]
    time = run_id.split("_")[2][:4]
    if group == "m1":
        if day == "20260925":
            return "Release Sep (SVM)"
        if day == "20261105":
            return "Control Nov 5 (SVM)"
        return "Ampliado Nov 6 (SVM)"
    if day == "20260920":
        return "Release Sep (CatBoost)"
    return f"Nov {int(day[-2:])} {time[:2]}:{time[2:]}"


def build_model_selection(context, styles) -> list:
    """Only the demonstration, and only its snapshotted factual evidence."""
    evidence = context.runtime.get("model_selection_evidence")
    if context.mode != "demonstration" or not evidence:
        return []
    m1_rows = evidence["model_1_runs"]
    m2_rows = evidence["model_2_runs"]
    paired = evidence["model_1_paired"]
    shared = paired["common"]
    high = paired["high_lithium"]
    ci = evidence["model_2_rf_delta_ci"]
    story = [
        PageBreak(),
        paragraph(
            "Anexo cientifico - seleccion de modelos para el siguiente release",
            styles["LS_H1"],
        ),
        paragraph(
            "Este anexo es parte del reporte que genera LithiumScope en la "
            "demostracion integrada real. Las cifras se extraen de los "
            "entrenamientos y sus tablas conservadas, no se inventan ni "
            "sustituyen las predicciones efectivamente calculadas. El "
            "empaquetado o la activacion siguen pendientes de aprobacion.",
            styles["LS_Body"],
        ),
        paragraph("A. Comparacion de generaciones de Modelo 1", styles["LS_H2"]),
    ]
    tab = [["Version M1", "Muestras", "RMSE global", "MAE global",
            "RMSE folds", "Estado"]]
    for entry in m1_rows:
        tab.append([
            _run_label(entry["run_id"], "m1"),
            str(entry["samples"]),
            _number(entry["metrics"].get("rmse"), 2),
            _number(entry["metrics"].get("mae"), 2),
            _number(entry.get("rmse_mean"), 2),
            entry["status"],
        ])
    story.append(table(tab, widths=[4.2*cm, 1.7*cm, 2.4*cm,
                                    2.2*cm, 2.3*cm, 3.2*cm], font_size=7))
    story.append(paragraph(
        "Cautela: el RMSE global del SVM ampliado no se compara directamente "
        "con el historico, pues sus 3.420 observaciones incorporan Li extremo "
        "hasta 4.160 ppm, distinto del maximo de 597,1 ppm en las 1.120 "
        "historicas. Son distribuciones objetivo distintas. La seleccion "
        "se sustenta principalmente en los 1.004 casos fisicos comunes.",
        styles["LS_Small"],
    ))
    story.append(paragraph("Evaluacion pareada de M1 sobre casos comunes",
                           styles["LS_H2"]))
    paired_rows = [["Error (menor es mejor)", "Historico", "Ampliado", "Mejora"]]
    for metric, label in (("rmse", "RMSE global (ppm)"),
                          ("mae", "MAE global (ppm)")):
        a = float(shared["old"][metric])
        b = float(shared["new"][metric])
        paired_rows.append([label, _number(a,2), _number(b,2),
                            _pct((a-b)/a)])
    paired_rows.append([
        "RMSE, Li alto Q75 (ppm)",
        _number(high["old"]["rmse"], 2),
        _number(high["new"]["rmse"], 2),
        _pct((high["old"]["rmse"]-high["new"]["rmse"]) /
             high["old"]["rmse"]),
    ])
    story.append(table(paired_rows,
                       widths=[5.2*cm,3.3*cm,3.3*cm,4.2*cm],font_size=7.5))
    story.append(paragraph(
        f"La comparacion pareada cubre {paired['matched']} muestras comunes; "
        f"{paired['different_outer_fold']} cambiaron de fold externo. "
        "Los IC95 agrupados de la ganancia RMSE y MAE son exploratorios "
        "y no sustituyen un holdout independiente. Para Li alto se "
        "mantiene subprediccion de concentraciones extremas.",
        styles["LS_Small"],
    ))
    story.append(paragraph(
        "Decision M1: usar el SVM ampliado de noviembre como candidato "
        "para el proximo release, por mejor error en los casos comunes, "
        "mayor cobertura y mejor comportamiento relativo en Li alto. "
        "Mantener documentada la limitacion de generalizacion y evaluar "
        "independientemente antes de afirmar superioridad universal.",
        styles["LS_Body"],
    ))
    story.extend([
        PageBreak(),
        paragraph("B. Historial de entrenamientos de Modelo 2",
                  styles["LS_H2"]),
        paragraph(
            "Los rankings siguientes muestran las metricas archivadas "
            "de cada entrenamiento (medias de folds espaciales); las "
            "cohortes 706 y 783 y el release original NO comparten "
            "automaticamente el mismo protocolo ni distribucion de "
            "validacion. No interpretar la tabla como test pareado.",
            styles["LS_Small"],
        ),
    ])
    rows = [["Version M2", "n", "feat.", "Algoritmo",
             "AUC folds", "AP folds", "Brier folds", "Estado"]]
    for entry in m2_rows:
        version = entry.get("feature_extractor_version") or 3
        feat = 170 if version == 4 else 126
        rows.append([
            _run_label(entry["run_id"], "m2"),
            str(entry["samples"]),
            str(feat),
            entry["algorithm"],
            _number(entry.get("roc_auc_mean"), 3),
            _number(entry.get("average_precision_mean"), 3),
            _number(entry.get("brier_score_mean"), 3),
            entry["status"],
        ])
    story.append(table(rows,widths=[3.3*cm,1.1*cm,1.1*cm,2.2*cm,
                                   2.0*cm,1.9*cm,2.1*cm,2.3*cm],
                       font_size=6.5))
    story.append(Spacer(1,8))
    story.append(paragraph(
        "Release original: CatBoost publicado con 126 caracteristicas "
        "y estado promovido. La version v4 contiene 706 muestras y "
        "170 caracteristicas, con Random Forest ganador, pero esta "
        "registrada como experimental y sin validacion externa "
        "independiente. Mantener todos los seis algoritmos para "
        "futuros experimentos; no se elimina ninguna familia.",
        styles["LS_Body"],
    ))
    story.append(paragraph("Competicion v4: las seis familias",
                           styles["LS_H2"]))
    rows = [["Algoritmo","AUC folds","AP folds","Brier folds","Top-1/3"]]
    for item in evidence["model_2_170_six_algorithms"]:
        rows.append([
            item["algorithm"],
            _number(item.get("roc_auc_mean")),
            _number(item.get("average_precision_mean")),
            _number(item.get("brier_score_mean")),
            _pct(item.get("three_zone_top1_accuracy")),
        ])
    story.append(table(rows,widths=[4.4*cm,3*cm,3*cm,3*cm,2.6*cm],
                       font_size=7))
    return story + _comparison_and_decision(evidence,styles)


def _comparison_and_decision(evidence, styles):
    compared = {v["algorithm"]:v for v in
                evidence["model_2_same_706_comparison"]}
    old = compared["original_release_protocol_retrained"]
    new = compared["random_forest"]
    ci = evidence["model_2_rf_delta_ci"]
    spectral = evidence["model_2_spectral_identity"]
    rows = [["Evaluacion en 706 TEST comunes",
             "Protocolo release", "M2 v4", "Delta v4-release"]]
    for key, label in (
        ("roc_auc", "ROC-AUC OOF"),
        ("average_precision", "Average Precision OOF"),
        ("brier", "Brier OOF (menor mejor)"),
    ):
        a,b=float(old[key]),float(new[key])
        rows.append([label,_number(a,4),_number(b,4),_number(b-a,4)])
    return [
        PageBreak(),
        paragraph("C. Comparacion M2 sobre el mismo conjunto historico",
                  styles["LS_H2"]),
        paragraph(
            "El CatBoost de septiembre se REENTRENO con su protocolo "
            "anterior sobre los mismos casos y folds que el candidato "
            "v4. No equivale a evaluar el archivo de release ya ajustado "
            "sobre un holdout externo nuevo.",
            styles["LS_Small"],
        ),
        table(rows,widths=[5.4*cm,3.6*cm,3.2*cm,3.8*cm],font_size=7.6),
        paragraph(
            "Intervalos de confianza del delta v4-release, bootstrap "
            "por grupos espaciales (95 %): "
            f"AUC [{_number(ci['auc_delta_ci95_low'],4)}, "
            f"{_number(ci['auc_delta_ci95_high'],4)}]; "
            f"AP [{_number(ci['ap_delta_ci95_low'],4)}, "
            f"{_number(ci['ap_delta_ci95_high'],4)}]. "
            "Ambos incluyen cero: no existe demostracion concluyente "
            "de mejora general.",
            styles["LS_Body"],
        ),
        paragraph("D. Diagnosticos que limitan la generalizacion",
                  styles["LS_H2"]),
        table([
            ["Indicador de auditoria v4", "Valor confirmado"],
            ["Muestras historicas / features", "706 / 170"],
            ["Elegibles QA95 solo TRAIN", "690 / 706"],
            ["Firmas espectrales distintas",
             str(spectral["distinct_exact_signatures_170"])],
            ["Filas redundantes por espectro",
             str(spectral["excess_redundant_rows"])],
            ["Firmas con etiquetas Li contrapuestas",
             str(spectral["mixed_fixed_class_signatures"])],
            ["Errores minimos por firma identica",
             str(spectral["minimum_classification_errors_due_only_to_identical_X"])],
        ],widths=[9.1*cm,6.9*cm],font_size=7.7),
        paragraph(
            "Las diferencias de etiqueta dentro de un mismo parche "
            "pueden representar rocas distintas; no prueban que el "
            "laboratorio este equivocado. La extrapolacion a sectores "
            "no vistos y los umbrales optimizados en OOF siguen siendo "
            "limitaciones importantes.",
            styles["LS_Small"],
        ),
        paragraph("E. Decision de seleccion para este reporte",
                  styles["LS_H2"]),
        paragraph(
            "M1: SVM ampliado de 3.420 observaciones. "
            "M2: CatBoost historico publicado, de 126 caracteristicas. "
            "La v4 es el candidato de investigacion con mejores "
            "metricas de desarrollo, pero su superioridad no es "
            "estadisticamente concluyente y la prediccion operativa "
            "actual no incorpora sus 44 descriptores red-edge: "
            "sustituirlos por ceros produciria scores incorrectos. "
            "Por eso NO se usa v4 para generar las predicciones "
            "de esta demostracion.",
            styles["LS_Body"],
        ),
        paragraph(
            "El PDF que sigue es un reporte de predicciones reales "
            "emitidas por estos dos artefactos versionados. Los diez "
            "casos de la demostracion tienen limitaciones de "
            "independencia y aplicabilidad. La seleccion aun necesita "
            "aprobacion explicita antes de empaquetar o activar el release.",
            styles["LS_Small"],
        ),
    ]
