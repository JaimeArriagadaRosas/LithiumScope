from __future__ import annotations

import os


HEADLESS_BACKEND = "Agg"


def configure_headless_matplotlib() -> str:
    """Force a non-GUI Matplotlib backend for the CLI application."""
    os.environ["MPLBACKEND"] = HEADLESS_BACKEND

    import matplotlib

    matplotlib.use(HEADLESS_BACKEND, force=True)
    backend = str(matplotlib.get_backend())
    if backend.lower() != HEADLESS_BACKEND.lower():
        raise RuntimeError(
            "LithiumScope requiere un backend Matplotlib no interactivo. "
            f"Esperado={HEADLESS_BACKEND}, actual={backend}."
        )
    return backend


def current_backend() -> str:
    import matplotlib

    return str(matplotlib.get_backend())
