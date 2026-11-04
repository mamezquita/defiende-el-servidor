# Notas de la selección de v2 (para el informe)

- La ventaja de E4 sobre E3 es pequeña (0,6 puntos de recall, 61 ataques, sin intervalo de
  confianza); el criterio decisivo fue la política, y como ventaja secundaria el tamaño
  (1,1 MB frente a 104,9 MB) y la inferencia.
- La FPR de v2 en validación (4,87 %) está cerca del límite y el umbral se eligió entre
  4 x 91 combinaciones, así que en test puede superar el 5 %.
- E2 cambia log1p y class_weight a la vez; sus efectos no se pueden separar.
- mlflow.db y mlartifacts/ son locales; la verificación externa se hace con evidencias/mlflow/.
