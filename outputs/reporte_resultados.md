## Resultados

**Muestras de validacion:** 1000

### Accuracy

- Top-1: **79.00%**
- Top-3: **95.80%**

### Matriz de confusion

Guardada en: `matriz_confusion.png`

### Accuracy correcta/incorrecta (binaria)

- **86.90%** -- ¿el modelo distingue bien si la palabra se dijo bien o mal, más allá de si acierta el subtipo exacto de error?

### Confianza del modelo (umbral 40%)

- Predicciones con confianza suficiente para mostrarle un resultado al chico: **916/1000** (91.60%)
- Sin confianza suficiente (el server pediría repetir la toma): **84** (8.40%)

### Latencia de inferencia

- Promedio: **17.71 ms** (batch=1, cuda)
