from datetime import datetime

from lithiumscope.cli import model_release_command, predict_command, results_command, train_command, scientific_lab_command
from lithiumscope.cli.display import clear_screen, header, pause
from lithiumscope.cli.prompts import choose
from lithiumscope.core.logger import (
    configure_logging,
    current_error_log_path,
    flush_logging,
    get_logger,
)
from lithiumscope.core.paths import ensure_runtime_directories

logger = get_logger("cli")


def main() -> int:
    ensure_runtime_directories()
    configure_logging()
    logger.info("LithiumScope started")
    while True:
        clear_screen()
        header()
        print("1. Entrenar modelos")
        print("2. Realizar predicción")
        print("3. Métricas y resultados")
        print("4. Cargar modelo versionado")
        print("5. Laboratorio de simulaciones")
        print("0. Salir")
        choice = choose("\nSeleccione una opción: ", {"0", "1", "2", "3", "4", "5"})
        try:
            if choice == "1":
                clear_screen()
                if train_command.run() is not False:
                    pause()
            elif choice == "2":
                clear_screen()
                if predict_command.run() is not False:
                    pause()
            elif choice == "3":
                clear_screen()
                if results_command.run() is not False:
                    pause()
            elif choice == "4":
                clear_screen()
                if model_release_command.run() is not False:
                    pause()
            elif choice == "5":
                scientific_lab_command.run()
            else:
                logger.info("LithiumScope finished")
                return 0
        except Exception as exc:
            error_id = datetime.now().strftime("%Y%m%d-%H%M%S")
            logger.exception("Unhandled application error id=%s", error_id)
            flush_logging()
            print(f"\n[ERROR {error_id}] {exc}")
            print(f"Detalle técnico: {current_error_log_path()}")
            pause()
