from __future__ import annotations

from pathlib import Path
import tempfile

from lithiumscope.cli.prompts import choose
from lithiumscope.core.paths import MODELS_DIR
from lithiumscope.persistence.active_models import active_model_path
from lithiumscope.distribution.github_releases import (
    download_release_asset,
    list_model_releases,
    cached_model_releases,
)
from lithiumscope.distribution.installer import (
    install_release_bundle,
)
from lithiumscope.runtime.console_status import Spinner


def _size_mib(size: int) -> str:
    if not size:
        return "tamaño desconocido"
    return f"{size / 1048576:.2f} MiB"


def _local_install():
    from lithiumscope.distribution.local_archive import local_release
    from lithiumscope.core.logging.session import run_log
    raw=input("Ruta del ZIP oficial descargado (Enter para cancelar): ").strip().strip('"')
    if not raw:
        return False
    path=Path(raw)
    print("Use un SHA-256 obtenido de GitHub o de una fuente confiable independiente.")
    digest=input("SHA-256 esperado: ").strip()
    release, verified=local_release(path,digest)
    print(f"Bundle local verificado: {release.tag_name}")
    if input("¿Instalar y activar ambos modelos? [s/N]: ").strip().lower() not in {"s","si","sí"}:
        return False
    with run_log("installation","local_release"):
        result=install_release_bundle(path,release,asset_sha256=verified)
    print(f"[OK] {result.release_tag} | Registro: {result.receipt_path}")
    return True


def _recovery():
    print("Descarga manual oficial: https://github.com/JaimeArriagadaRosas/LithiumScope/releases")
    print("1. Instalar ZIP oficial local con SHA-256 confiable")
    print("0. Volver (modelos activos y candidatos sin cambios)")
    if choose("Seleccione [0-1]: ",{"1","0"})=="1":
        return _local_install()
    return False


def run() -> None:
    print("\nCARGAR MODELO VERSIONADO")
    spinner = Spinner(
        "Consultando modelos publicados en GitHub..."
    ).start()
    try:
        releases = list_model_releases()
        spinner.succeed(
            "Modelos publicados encontrados: "
            f"{len(releases)}"
        )
    except Exception as exc:
        spinner.fail("No fue posible consultar GitHub Releases")
        from lithiumscope.distribution.transport import explain_error
        from lithiumscope.core.logging.session import get_logger
        get_logger("distribution.installation").exception("Release catalog unavailable")
        print(explain_error(exc))
        try:
            releases, checked = cached_model_releases()
        except (OSError, ValueError, KeyError, TypeError):
            releases, checked = [], None
        if releases:
            print(f"[AVISO] Catálogo local del {checked}; no confirma la última versión remota.")
        else:
            return _recovery()

    if not releases:
        print(
            "\nNo hay GitHub Releases publicados que contengan "
            "un bundle de modelos compatible."
        )
        print(
            "El programa busca únicamente assets *-artifacts.zip; "
            "ignora los archivos Source code generados por GitHub."
        )
        return _recovery()

    print("Orden: publicación más reciente primero (según catálogo disponible).")
    for index, release in enumerate(
        releases,
        start=1,
    ):
        published = (
            release.published_at[:10]
            if release.published_at
            else "sin fecha"
        )
        print(
            f"{index}. {release.tag_name} | "
            f"{published} | "
            f"{_size_mib(release.asset_size)}"
        )
    local_option = str(len(releases) + 1)
    print(f"{local_option}. Instalar ZIP oficial local")
    print("0. Volver")

    allowed = {
        "0", local_option,
        *{
            str(index)
            for index in range(
                1,
                len(releases) + 1,
            )
        },
    }
    choice = choose(
        "\nSeleccione el modelo versionado: ",
        allowed,
    )
    if choice == "0":
        return False
    if choice == local_option:
        return _local_install()

    release = releases[
        int(choice) - 1
    ]
    print(f"\nBundle seleccionado: {release.tag_name}")
    print("Esta operación instalará y activará los modelos de este release.")
    for group in ("model_1", "model_2"):
        print(f"Modelo activo {group}: {active_model_path(group) or 'N/D'}")
    try:
        confirmation = input("¿Continuar? [s/N]: ").strip().lower()
    except EOFError:
        confirmation = ""
    if confirmation not in {"s", "sí", "si"}:
        print("[CANCELADO] No se descargaron ni activaron modelos.")
        return False
    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    from lithiumscope.core.logging.session import run_log,get_logger
    with run_log("installation","versioned_model"):
        get_logger("distribution.installation").info("Authorized release installation: %s",release.tag_name)
        with tempfile.TemporaryDirectory(
            prefix=".release_download_",
            dir=MODELS_DIR,
        ) as temporary:
            destination = (
                Path(temporary)
                / release.asset_name
            )
            download_spinner = Spinner(
                f"Descargando {release.asset_name}"
            ).start()

            def progress(
                downloaded: int,
                total: int,
            ) -> None:
                if total > 0:
                    download_spinner.update(
                        f"Descargando {release.asset_name}: "
                        f"{downloaded / 1048576:.1f}/"
                        f"{total / 1048576:.1f} MiB"
                    )
                else:
                    download_spinner.update(
                        f"Descargando {release.asset_name}: "
                        f"{downloaded / 1048576:.1f} MiB"
                    )

            try:
                archive, asset_sha256 = (
                    download_release_asset(
                        release,
                        destination,
                        progress_callback=progress,
                    )
                )
                download_spinner.succeed(
                    "Descarga verificada: "
                    f"{release.asset_name}"
                )
            except Exception as exc:
                download_spinner.fail(f"Falló la descarga: {release.asset_name}")
                from lithiumscope.distribution.transport import explain_error
                get_logger("distribution.installation").exception("Release download failed")
                print(explain_error(exc))
                print(f"Asset oficial: {release.asset_url}")
                if release.asset_digest:
                    print(f"SHA-256 esperado: {release.asset_digest}")
                return _recovery()

            install_spinner = Spinner(
                "Validando e instalando Modelos 1 y 2..."
            ).start()
            try:
                result = install_release_bundle(
                    archive,
                    release,
                    asset_sha256=asset_sha256,
                )
                install_spinner.succeed(
                    "Modelo versionado instalado: "
                    f"{result.release_tag}"
                )
            except Exception:
                install_spinner.fail(
                    "El bundle no pudo instalarse"
                )
                raise


    print(
        f"\nModelo cargado: "
        f"{result.release_tag}"
    )
    for model in result.models:
        print(
            f"  {model.model_group}: "
            f"{model.algorithm} | "
            f"{model.run_id}"
        )
    print(
        f"  Registro: "
        f"{result.receipt_path}"
    )
    print(
        "\nLas predicciones usarán estos "
        "artefactos como modelos activos."
    )
