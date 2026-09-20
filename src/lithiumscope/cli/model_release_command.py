from __future__ import annotations

from pathlib import Path
import tempfile

from lithiumscope.cli.prompts import choose
from lithiumscope.core.paths import MODELS_DIR
from lithiumscope.distribution.github_releases import (
    download_release_asset,
    list_model_releases,
)
from lithiumscope.distribution.installer import (
    install_release_bundle,
)
from lithiumscope.runtime.console_status import Spinner


def _size_mib(size: int) -> str:
    if not size:
        return "tamaño desconocido"
    return f"{size / 1048576:.2f} MiB"


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
    except Exception:
        spinner.fail(
            "No fue posible consultar GitHub Releases"
        )
        raise

    if not releases:
        print(
            "\nNo hay GitHub Releases publicados que contengan "
            "un bundle de modelos compatible."
        )
        print(
            "El programa busca únicamente assets *-artifacts.zip; "
            "ignora los archivos Source code generados por GitHub."
        )
        return

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
    print("0. Volver")

    allowed = {
        "0",
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
        return

    release = releases[
        int(choice) - 1
    ]
    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

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
        except Exception:
            download_spinner.fail(
                "Falló la descarga: "
                f"{release.asset_name}"
            )
            raise

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
