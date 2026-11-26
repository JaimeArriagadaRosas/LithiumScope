# Incidencia de la decisión integrada y alcance aprobado

En los PDF de 13/11 (release SVM/CatBoost) y 26/11 (RF/RF), página 2,
ambos modelos predijeron diez casos. El segundo reporte omitió la etapa
espectral porque el filtro exigía tres advertencias M1 iguales a OK.
El conteo marginal de página 6 era 1/10 y el diagnóstico final de M1
era 10/10 fuera de dominio, coherente con la capa multivariada adicional.
El ranking ignoró q90 aun estando disponible, porque no había filas OK.

Nueva política: distinguir validez y aplicabilidad, priorizar M1 OK y
completar hasta tres con salidas finitas no negativas OUT_OF_DOMAIN o
INSUFFICIENT_PROFILE. No aceptar entradas/coordenadas insuficientes o
predicciones físicamente inválidas. Si hay al menos dos M2 OK, comparar
ellos; en caso contrario comparar scores válidos en [0,1] como ranking
exploratorio. Empates y menos de dos scores válidos: sin ganador.
Siempre mostrar ambas etapas, con etiquetas breves y diagnósticos completos
en su sección. No sustituir las imágenes ausentes ni alterar scores.
No usar el Li observado para seleccionar; solo comprobación retrospectiva.

La demo del 26/11 tiene M1 RMSE 1.9461, MAE 1.4994, R2 .9301; M2 AUC .25,
AP .1964, BA .5, F1 0, 0 positivos. El release tenía M1 RMSE 3.2303, MAE 2.7460,
R2 .8074; M2 AUC .6875, AP .6429, BA .6875, F1 .5. No confundir resultados
de desarrollo con esta demostración de diez casos sin independencia certificada.
No ajustar modelos ni umbrales para estos casos. No publicar release por
completar las fases. El PDF deberá reproducirse localmente desde los artefactos
reales; los cambios aquí son de decisión posterior a inferencia y presentación.

## Validación del 26/11 — commit 571df98

Suite local: 1004 pruebas, 1000 aprobadas, 1 fallo, 0 errores, 3 omitidas.
Cobertura 57.02%, duración 126.453 s, identidad Git sin cambios.
Falló únicamente test_no_valid_inputs_still_has_stage_two: la rama de selección
vacía devolvía la página antes del encabezado de etapa 2.
Se corrigió la rama del renderizador, no el contrato de la prueba.
Se conserva el test original y se añade al test de PDF existente un cuarto
escenario: ningún M1 utilizable, etapa 2 en página 2, sin casos comparables y
sin ganador fabricado. Los casos con datos válidos mantienen su ranking.
La corrección requiere nueva ejecución; no implica entrenamiento ni publicación.
