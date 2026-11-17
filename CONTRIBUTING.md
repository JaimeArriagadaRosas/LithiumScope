# Guía de contribución

LithiumScope es un proyecto académico independiente desarrollado originalmente por **Jaime Arriagada Rosas**. Las propuestas externas pueden aportar mejoras sin modificar esa atribución.

## Antes de contribuir

1. Consultar [README.md](README.md), [arquitectura](docs/architecture/ARCHITECTURE.md) y [principios de diseño](docs/architecture/DESIGN_PRINCIPLES.md).
2. Proponer los cambios de alcance amplio mediante un issue antes de implementarlos.
3. No incluir credenciales, datos restringidos, pesos sin procedencia verificable ni resultados científicos atribuidos incorrectamente.
4. No modificar releases publicados, modelos activos o evidencia histórica como efecto secundario de cambios de presentación.

## Preparación local

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pytest -q
python -m ruff check src tests main.py
python scripts/check_code_size.py --check
```

Los extras `ml` e `imagery` de `pyproject.toml` son necesarios para determinados modelos y datos satelitales. Algunas pruebas con artefactos privados se omiten en checkouts limpios; esto no constituye validación científica independiente.

## Criterios de cambio

- Mantener una aplicación modular: CLI delgada, responsabilidades separadas y contratos explícitos.
- Preservar la independencia científica de M1 (regresión) y M2 (clasificación/prospectividad).
- Escribir o actualizar tests cuando se alteren contratos o comportamientos.
- Documentar datos, `run_id`, hashes, particiones de validación, límites y resultados negativos.
- Evitar leakage, métricas incomparables y afirmaciones de validación externa sin evidencia.
- No crear tags, activar modelos ni publicar paquetes automáticamente.

## Pull requests y licencia

Describir objetivo, archivos modificados, pruebas, riesgos y procedencia de datos. La incorporación de una contribución requiere aceptación del mantenedor. Las contribuciones aceptadas se distribuirán bajo la [licencia del repositorio](LICENSE), salvo acuerdo escrito diferente. La atribución original permanece especificada en [CREDITS.md](CREDITS.md).

Para vulnerabilidades, consultar [SECURITY.md](SECURITY.md).
