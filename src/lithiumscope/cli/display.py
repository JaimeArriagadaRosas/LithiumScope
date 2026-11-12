from __future__ import annotations

import os
import sys


def clear_screen() -> None:
    """Clear interactive terminal screens without erasing log files or redirected output."""
    if not sys.stdout.isatty():
        return
    if os.name == "nt":
        os.system("cls")
    elif os.environ.get("TERM") not in (None, "", "dumb"):
        os.system("clear")

from lithiumscope.core.device import DeviceInfo


def header() -> None:
    print("\n" + "=" * 52)
    print("                    LITHIUMSCOPE")
    print("=" * 52)


def device_summary(info: DeviceInfo) -> None:
    print(f"\nDispositivo: {info.accelerator.upper()} — {info.name}")
    print(f"CPU threads: {info.cpu_count}")


def pause() -> None:
    try:
        input("\nPresione Enter para continuar...")
    except EOFError:
        pass
