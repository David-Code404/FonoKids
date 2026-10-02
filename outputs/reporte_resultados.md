## Resultados

**Muestras de validacion:** 3168

### Accuracy

- Top-1: **70.27%**
- Top-3: **92.01%**

### Matriz de confusion

Guardada en: `matriz_confusion.png`

### Accuracy correcta/incorrecta (binaria)

- **78.16%** -- ¿el modelo distingue bien si la palabra se dijo bien o mal, más allá de si acierta el subtipo exacto de error?

### Confianza del modelo (umbral 40%)

- Predicciones con confianza suficiente para mostrarle un resultado al chico: **2780/3168** (87.75%)
- Sin confianza suficiente (el server pediría repetir la toma): **388** (12.25%)

### Latencia de inferencia

- Promedio: **15.38 ms** (batch=1, cuda)
