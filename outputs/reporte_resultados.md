## Resultados

**Muestras de validacion:** 140

### Accuracy

- Top-1: **67.86%**
- Top-3: **95.71%**

### Matriz de confusion

Guardada en: `matriz_confusion.png`

### Accuracy correcta/incorrecta (binaria)

- **77.86%** -- ¿el modelo distingue bien si la palabra se dijo bien o mal, más allá de si acierta el subtipo exacto de error?

### Confianza del modelo (umbral 40%)

- Predicciones con confianza suficiente para mostrarle un resultado al chico: **140/140** (100.00%)
- Sin confianza suficiente (el server pediría repetir la toma): **0** (0.00%)

### Latencia de inferencia

- Promedio: **5.62 ms** (batch=1, cuda)
