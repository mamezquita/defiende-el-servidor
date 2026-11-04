# Notas de la evaluación en test (para el informe)

No se cambió nada después de ver estos resultados: ni el modelo, ni el umbral, ni la selección.
Las cifras salen de `metricas-test.json` y `desplazamiento-test.json`. Ese análisis es descriptivo y
posterior a la evaluación única; muestra asociaciones, no prueba causas.

## Qué pasó

| Métrica (umbral 0,64) | Validación | Test | Cambio |
|---|---:|---:|---:|
| FPR | 4,87 % | 18,80 % | peor (+13,93 puntos) |
| Precision de ataques | 94,73 % | 86,34 % | peor (-8,39 puntos) |
| Macro F1 | 92,77 % | 89,60 % | peor (-3,17 puntos) |
| Accuracy | 92,78 % | 89,90 % | peor (-2,88 puntos) |
| Recall de ataques | 90,35 % | 96,99 % | mejor (+6,64 puntos) |

**v2 no cumple en test el límite de FPR <= 5 % con el que se eligió.** El modelo marca más conexiones
como ataque que en validación: por eso suben a la vez el recall y las falsas alarmas.

## Causas posibles, de más a menos respaldada

### 1. Desplazamiento de distribución: es la explicación con más respaldo

El tráfico normal de test no se parece al de train, mientras que validación sí, porque sale del
mismo archivo. Comparando filas únicas, con la distancia de Kolmogorov-Smirnov en 39 variables
numéricas:

| Normales frente a train | Mediana | Máximo |
|---|---:|---:|
| Validación (control) | 0,007 | 0,011 |
| Test | 0,131 | 0,276 |

- 35 de las 39 variables superan en test el máximo del control. Las más desplazadas son `sttl`,
  `ct_state_ttl`, `dload`, `ackdat` y `dmean`.
- Los ataques también se desplazan, aunque menos: mediana de 0,071 frente a 0,015 del control.
- Las conexiones normales con `state = INT` pasan de 5,0 % en train a 12,1 % en test, y el 53,9 % de
  ellas se marca como ataque. La tasa en `FIN` es 18,7 % y en `CON` es 0,1 %.
- Las categorías nuevas no son la causa: solo 5 filas de test tienen un valor de `state` no visto en
  train.

### 2. Umbral elegido en datos que no tienen ese desplazamiento

Validación sale del archivo de training, así que no podía anticipar este cambio. Además, la FPR de
validación estaba cerca del límite y el umbral se eligió entre 4 x 91 combinaciones. Ese optimismo de
selección existe, pero por su tamaño probable no explica por sí solo una subida de casi 14 puntos.

### 3. Duplicados: explican parte de la mejora del recall, no la FPR

El test se evaluó completo, sin deduplicar, y el 56,4 % de sus filas de ataque son duplicados
sobrantes, frente al 7,6 % de las normales.

| Test | Recall | FPR | Precision |
|---|---:|---:|---:|
| Completo | 96,99 % | 18,80 % | 86,34 % |
| Solo filas únicas | 93,87 % | 17,98 % | 75,09 % |

Sin duplicados, el recall baja 3,13 puntos y la FPR casi no cambia. La precision baja mucho porque
los ataques pasan de 55,1 % a 36,6 % de las filas.

### 4. Solapamiento con train y validación: no explica la FPR

| Filas de test | Cantidad | Recall | FPR |
|---|---:|---:|---:|
| Con entradas vistas en train o validación | 8.313 | 99,83 % | 19,52 % |
| Con entradas no vistas | 74.019 | 96,47 % | 18,77 % |

La FPR es casi igual en ambos grupos. Las filas vistas sí tienen un recall más alto, pero son solo el
10,1 % del test, y las no vistas ya alcanzan 96,47 %.

### 5. Proporción de clases: no explica la caída de la precision

Con el recall y la FPR de validación, la proporción de ataques de test daría una precision de 95,79 %,
mayor que en validación. La precision medida es 86,34 %: la caída viene de la FPR.

## Lectura para el informe

- La selección y la evaluación fueron válidas y trazables: este resultado es lo que ese proceso
  protege, porque el test no intervino en ninguna decisión.
- La FPR de validación **no fue una buena estimación** para datos del archivo de test. La cifra para
  informar es la de test.
- Corregir el umbral con estos resultados invalidaría la evaluación. Una mejora futura necesitaría
  datos nuevos o una validación que represente este desplazamiento.
- Solo se evaluó v2, así que estos datos no permiten comparar v1 y v2 en test.
