from __future__ import annotations

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
