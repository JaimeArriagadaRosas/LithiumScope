"""Published academic reference, distinct from any local OOF evaluation."""

OASIS_M1 = {
    "source": "Alarcón Arias y Saud Rauld (2026), Predicción de concentraciones de litio mediante modelos de aprendizaje automático aplicados a datos geoquímicos y geoespaciales, secciones 7.3, 8.4, 8.6 y 9.1.1.",
    "algorithm": "Random Forest",
    "rows": 681,
    "rmse_mean": 5.361,
    "rmse_std": .595,
    "r2_mean": .499,
    "r2_std": .086,
    "mae": None,
    "protocol": "KFold aleatorio anidado 5x5; shuffle=True; random_state=42; 100 trials Optuna por fold externo; percentiles 2.5–97.5; balance de óxidos 94–102.",
    "original_predictions_available": False,
    "direct_comparability": False,
}
