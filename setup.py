"""Build-time packaging of canonical YAML/JSON defaults.

The source of truth remains the repository's config/ directory. This hook
copies those files into the installed package without adding a second
maintained copy to the repository.
"""

from pathlib import Path
import shutil

from setuptools import setup
from setuptools.command.build_py import build_py


class BuildPyWithDefaults(build_py):
    def run(self) -> None:
        super().run()
        source = Path(__file__).resolve().parent / "config"
        destination = Path(self.build_lib) / "lithiumscope" / "_defaults"
        destination.mkdir(parents=True, exist_ok=True)
        for path in sorted(source.iterdir()):
            if path.is_file() and path.suffix.lower() in {".yaml", ".json"}:
                shutil.copy2(path, destination / path.name)

    def get_outputs(self, include_bytecode: int = 1) -> list[str]:
        outputs = super().get_outputs(include_bytecode=include_bytecode)
        source = Path(__file__).resolve().parent / "config"
        destination = Path(self.build_lib) / "lithiumscope" / "_defaults"
        if source.is_dir():
            outputs.extend(
                str(destination / path.name)
                for path in sorted(source.iterdir())
                if path.is_file() and path.suffix.lower() in {".yaml", ".json"}
            )
        return outputs


setup(cmdclass={"build_py": BuildPyWithDefaults})
