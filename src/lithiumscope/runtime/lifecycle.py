from __future__ import annotations

from collections.abc import Callable

from lithiumscope.core.logger import get_logger
from lithiumscope.core.run_resume import recover_abandoned_runs
from lithiumscope.core.visualization import configure_headless_matplotlib
from lithiumscope.runtime.graceful_shutdown import GracefulExit, get_shutdown_manager
from lithiumscope.runtime.preboot import run_preboot

logger = get_logger("runtime.lifecycle")


def run_application(entrypoint: Callable[[], int]) -> int:
    manager = get_shutdown_manager()
    manager.install()

    try:
        backend = configure_headless_matplotlib()
        recovered = recover_abandoned_runs()
        if recovered:
            logger.warning(
                "Recovered %d abandoned training run(s) as crashed: %s",
                len(recovered),
                ", ".join(path.name for path in recovered),
            )
        logger.info("Matplotlib backend initialized: %s", backend)

        report = run_preboot(verbose=True)
        if not report.core_ready:
            logger.error("Core preboot failed; application will not start")
            print("\nPreboot falló en requisitos esenciales. Revise el reporte indicado arriba.")
            return 2
        return int(entrypoint())
    except GracefulExit as exc:
        return int(exc.code or 0)
    except KeyboardInterrupt:
        manager.request_shutdown("keyboard_interrupt", exit_code=130)
        return 130
    except Exception:
        logger.exception("Fatal error escaped the application boundary")
        manager.request_shutdown("fatal_error", exit_code=1)
        return 1
    finally:
        manager.finalize()
