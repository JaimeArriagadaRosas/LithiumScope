from datetime import datetime

from lithiumscope.cli import model_release_command, predict_command, results_command, train_command
from lithiumscope.cli.display import header, pause
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
        header()
        print("1. Entrenar modelos")
        print("2. Realizar predicción")
        print("3. Métricas y resultados")
        print("4. Cargar modelo versionado")
        print("0. Salir")
        choice = choose("\nSeleccione una opción: ", {"0", "1", "2", "3", "4"})
        try:
            if choice == "1":
                train_command.run()
                pause()
            elif choice == "2":
                predict_command.run()
                pause()
            elif choice == "3":
                results_command.run()
                pause()
            elif choice == "4":
                model_release_command.run()
                pause()
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
