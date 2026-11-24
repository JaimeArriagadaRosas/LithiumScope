# Desarrollo posterior al primer push

Rama de trabajo: feat/integral-scientific-refactor. No fusionar en main ni publicar releases.
La primera ejecucion CI corresponde a b0c385788bb48b1c5079a4a1c9bfbad17ac019c2.
Durante ediciones intermedias, incluir [skip ci] en los mensajes de commit; no deshabilitar Actions globalmente. Al terminar, eliminar la opcion temporal, ejecutar una regresion local y un commit de validacion sin skip ci. No atribuir resultados de CI a commits posteriores no evaluados.

## Diagnostico temporal 5.4

Tras git pull de la rama de trabajo, iniciar main.py y elegir 5 > 4.
La opción ejecuta sintaxis, Ruff, arquitectura y pytest completo con cobertura, y después el inventario local. El reporte se guarda en results/diagnostics/development_.../development_report.json, con resumen en consola.
Comprueba SHA y estado de archivos esperados, versiones, recursos, arquitectura, modelos activos sin deserializar,
ultima suite local (cobertura e identidad si consta), preparacion M1 real, particiones espaciales, QA de M2 y red-edge
offline de la cohorte congelada. No descarga datos, no ajusta estimadores, no regenera cache ni PDF, no activa modelos.
La revision V4 verifica fuentes y alineacion, pero no reextrae los 126 descriptores base; el informe no es certificacion
de rendimiento. Los fallos de una seccion no impiden conservar otras. El detalle de excepciones queda solo en los logs
locales; no se incluyen datos crudos, logs completos, variables de entorno, credenciales ni binarios.
Enviar primero el JSON revisado, no todos los logs ni los datos. La opción 5 > 4 incluye la suite nueva, con identificación del commit y detalle de errores por módulo;
la cobertura anterior no se vincula automaticamente al commit actual. Se eliminaran el submenu y su paquete temporal
antes del cierre definitivo.

## Cierre
1. Revisar evidencia real local y distinguir verificaciones de codigo, faltantes de datos y limitaciones experimentales.
2. Completar pruebas y correcciones de preparacion, reporte PDF, variantes controladas y diagnosticos de aprendizaje.
3. Preparar experimentos de las seis familias con folds fijos, sin publicar modelos ni inferir validacion externa.
4. Ejecutar pruebas de Linux/Windows y Python 3.11/3.12/3.13 tras eliminar el diagnostico temporal.
