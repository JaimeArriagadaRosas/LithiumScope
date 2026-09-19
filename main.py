import os

# LithiumScope is a CLI/batch application: force headless plotting before
# any transitive Matplotlib import can select a GUI backend such as TkAgg.
os.environ["MPLBACKEND"] = "Agg"

from lithiumscope.cli.menu import main
from lithiumscope.runtime.lifecycle import run_application

if __name__ == "__main__":
    raise SystemExit(run_application(main))
