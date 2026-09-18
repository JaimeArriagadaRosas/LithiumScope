from lithiumscope.cli.prompts import choose
from lithiumscope.results.browser import (
    list_runs,
    open_latest_excel,
    open_results_folder,
    preview_latest,
)


def _preview(group: str) -> None:
    dashboard = preview_latest(group)
    print(f"Dashboard abierto: {dashboard}" if dashboard else f"No hay ejecuciones para {group}.")


def _excel(group: str) -> None:
    path = open_latest_excel(group)
    print(f"Excel abierto: {path}" if path else f"No hay Excel de resultados para {group}.")


def run() -> None:
    print("\nCENTRO DE MÉTRICAS Y RESULTADOS")
    print("1. Previsualizar dashboard — Modelo 1")
    print("2. Previsualizar dashboard — Modelo 2")
    print("3. Abrir Excel — Modelo 1")
    print("4. Abrir Excel — Modelo 2")
    print("5. Listar ejecuciones")
    print("6. Abrir carpeta completa de resultados")
    print("0. Volver")
    choice = choose("> ", {str(i) for i in range(7)})

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
            print("  Sin ejecuciones." if not runs else "")
            for run_dir in runs[:10]:
                print(f"  - {run_dir.name} -> {run_dir}")
    elif choice == "6":
        open_results_folder()
