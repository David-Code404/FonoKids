## Resultados

**Muestras de validacion:** 2543

### Accuracy

- Top-1: **68.11%**
- Top-3: **94.57%**

### Matriz de confusion

Guardada en: `matriz_confusion.png`

### Accuracy correcta/incorrecta (binaria)

- **76.09%** -- ¿el modelo distingue bien si la palabra se dijo bien o mal, más allá de si acierta el subtipo exacto de error?

### Confianza del modelo (umbral 40%)

- Predicciones con confianza suficiente para mostrarle un resultado al chico: **2171/2543** (85.37%)
- Sin confianza suficiente (el server pediría repetir la toma): **372** (14.63%)

### Latencia de inferencia

- Promedio: **5.55 ms** (batch=1, cuda)
