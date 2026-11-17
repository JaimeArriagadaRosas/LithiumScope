# Seguridad

## Alcance del soporte

LithiumScope es una aplicación local y académica. La rama `main` es el código que recibe mantenimiento; no se garantiza soporte de seguridad para releases y modelos históricos.

## Reporte responsable

**No publique vulnerabilidades explotables, credenciales o datos personales en un issue público.** Comuníquelas de forma privada al propietario del repositorio a través de las funciones de contacto de su perfil GitHub. En caso de habilitarse Private Vulnerability Reporting en el repositorio, utilice preferentemente esa opción.

Incluya versión o commit, entorno, reproducción mínima, impacto y evidencia suficiente; evite compartir datasets no autorizados.

## Riesgos conocidos de uso

- **Modelos `.joblib`**: la deserialización puede ejecutar código. Cargue solamente artefactos de fuentes que considere confiables. Un hash de integridad no sustituye la confianza en el origen.
- **Entradas externas**: verifique permisos y procedencia de CSV, Excel, GeoTIFF, NPY y archivos ZIP.
- **Descargas y dependencias**: utilice entornos virtuales y revise los endpoints, paquetes y archivos instalados.
- **Datos científicos**: los resultados y scores exploratorios no sustituyen validaciones geológicas ni análisis de riesgo.

Esta política no implica que el proyecto tenga una auditoría de seguridad formal.
