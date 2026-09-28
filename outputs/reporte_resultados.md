## Resultados

**Muestras de validacion:** 2543

### Accuracy

- Top-1: **66.93%**
- Top-3: **94.42%**

### Matriz de confusion

Guardada en: `matriz_confusion.png`

### Accuracy correcta/incorrecta (binaria)

- **74.68%** -- ¿el modelo distingue bien si la palabra se dijo bien o mal, más allá de si acierta el subtipo exacto de error?

### Confianza del modelo (umbral 40%)

- Predicciones con confianza suficiente para mostrarle un resultado al chico: **2167/2543** (85.21%)
- Sin confianza suficiente (el server pediría repetir la toma): **376** (14.79%)

### Latencia de inferencia

- Promedio: **21.64 ms** (batch=1, cuda)
