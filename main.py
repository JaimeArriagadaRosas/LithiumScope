import os
from pathlib import Path
import sys

# Allow a source checkout to bootstrap itself before the editable package is
# installed or updated.
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# LithiumScope is a CLI/batch application: force headless plotting before
# any transitive Matplotlib import can select a GUI backend such as TkAgg.
os.environ["MPLBACKEND"] = "Agg"

from lithiumscope.runtime.dependencies import ensure_runtime_dependencies

# Repair declared runtime dependencies before importing the rest of the app.
# This lets a normal "python main.py" recover after a git pull adds a package.
dependency_repair = ensure_runtime_dependencies(verbose=True)
if not dependency_repair.success:
    missing = ", ".join(dependency_repair.missing_after) or "desconocidas"
    print(
        "\nNo fue posible completar las dependencias requeridas: "
        + missing
    )
    print(
        "Revise la conexión a Internet, permisos del entorno virtual "
        "y vuelva a ejecutar: python main.py"
    )
    raise SystemExit(2)

from lithiumscope.cli.menu import main
from lithiumscope.runtime.lifecycle import run_application

if __name__ == "__main__":
    raise SystemExit(run_application(main))
