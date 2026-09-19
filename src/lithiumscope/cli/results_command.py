from lithiumscope.cli.prompts import choose
from lithiumscope.results.browser import (
    list_runs,
    open_latest_excel,
    open_results_folder,
    preview_latest,
)
from lithiumscope.results.catalog import run_catalog
from lithiumscope.results.release import build_release_candidate_manifest


def _preview(group: str) -> None:
    dashboard = preview_latest(group)
    print(
        f"Dashboard abierto: {dashboard}"
        if dashboard
        else f"No hay ejecuciones para {group}."
    )


def _excel(group: str) -> None:
    path = open_latest_excel(group)
    print(
        f"Excel abierto: {path}"
        if path
        else f"No hay Excel de resultados para {group}."
    )


def _compare(group: str) -> None:
    catalog = run_catalog(group)
    if catalog.empty:
        print(f"No hay ejecuciones para {group}.")
        return
    columns = [
        "run_id",
        "state",
        "winner",
        "primary_metric_value",
        "failed_algorithms",
        "release_candidate",
    ]
    print(catalog[columns].head(20).to_string(index=False))


def run() -> None:
    print("\nCENTRO DE MÉTRICAS Y RESULTADOS")
    print("1. Previsualizar dashboard — Modelo 1")
    print("2. Previsualizar dashboard — Modelo 2")
    print("3. Abrir Excel — Modelo 1")
    print("4. Abrir Excel — Modelo 2")
    print("5. Listar ejecuciones")
    print("6. Abrir carpeta completa de resultados")
    print("7. Comparar ejecuciones — Modelo 1")
    print("8. Comparar ejecuciones — Modelo 2")
    print("9. Evaluar candidato local para futuro tag")
    print("0. Volver")
    choice = choose("> ", {str(i) for i in range(10)})

    if choice == "1":
        _preview("model_1")
    elif choice == "2":
        _preview("model_2")
    elif choice == "3":
        _excel("model_1")
    elif choice == "4":
        _excel("model_2")
    elif choice == "5":
        for group in ("model_1", "model_2"):
            print(f"\n{group.upper()}")
            runs = list_runs(group)
            if not runs:
                print("  Sin ejecuciones.")
            for run_dir in runs[:10]:
                print(f"  - {run_dir.name} -> {run_dir}")
    elif choice == "6":
        open_results_folder()
    elif choice == "7":
        _compare("model_1")
    elif choice == "8":
        _compare("model_2")
    elif choice == "9":
        try:
            path = build_release_candidate_manifest()
            print(f"Candidato local generado: {path}")
            print("No se creó ningún tag de GitHub.")
        except RuntimeError as exc:
            print(f"No hay candidato válido todavía: {exc}")
