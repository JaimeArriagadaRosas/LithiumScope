# Instalación y ejecución

## Requisitos

- Python **3.11 o superior** (GitHub CI verifica Python 3.11, 3.12 y 3.13).
- Entorno virtual recomendado, espacio para datasets y conexión a Internet cuando se descarguen recursos.
- Los modelos pesados y herramientas satelitales se habilitan mediante extras de `pyproject.toml`.

## Windows, desde un clon

```powershell
git clone https://github.com/JaimeArriagadaRosas/LithiumScope.git
cd LithiumScope
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[ml,imagery]"
python main.py
```

Para desarrollo y pruebas, instalar `.[ml,imagery,dev]`. En Linux, activar con `source .venv/bin/activate`.

## Entradas

- `python main.py`: ejecuta el arranque de un checkout, con comprobación de dependencias y menú.
- `lithiumscope`: entrada registrada en `pyproject.toml` tras instalar el paquete.
- `lithiumscope-inspect`: inspección del pipeline de adquisición y fuentes sin entrenar.

El menú ofrece entrenamiento, predicción, métricas, carga de modelos versionados, laboratorio de simulaciones y salida.

**Si se instala desde wheel fuera de un clon**, LithiumScope guarda configuración, datos, modelos y resultados en `~/.lithiumscope/` (o la ubicación indicada en `LITHIUMSCOPE_HOME`). Los YAML y JSON predeterminados se incluyen en la distribución y solo se copian si aún no existen; no se sobrescriben archivos modificados. Al ejecutarlo desde el checkout sin esa variable, mantiene las rutas propias del repositorio.

## Primer uso sin entrenar

1. Ejecutar el programa y seleccionar **4. Cargar modelo versionado**.
2. Seleccionar un release que contenga el asset `*-artifacts.zip` y verificar la instalación.
3. Volver al menú y seleccionar **2. Realizar predicción**.

El preboot no requiere descargar todos los datasets antes de mostrar el menú; la preparación de fuentes se inicia cuando corresponde. **Los modelos activos de un equipo son locales** y pueden no coincidir con el release más reciente.

## Validación de instalación

```powershell
python -m pip check
python -m pytest -q
python -m ruff check src tests main.py
python scripts/check_code_size.py --check
```

Algunas pruebas científicas requieren datos y pesos excluidos de Git. Un checkout limpio puede reportar omisiones justificadas.

## Advertencias

Los `.joblib` deben proceder de fuentes confiables; deserializarlos puede ejecutar código. El instalador comprueba hashes y estructura del ZIP, pero eso no sustituye la confianza en el origen. La documentación de empaquetado y compatibilidad está en [PACKAGING.md](PACKAGING.md).
