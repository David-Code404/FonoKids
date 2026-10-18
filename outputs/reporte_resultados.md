## Resultados

**Muestras de validacion:** 3064

### Accuracy

- Top-1: **67.17%**
- Top-3: **92.46%**

### Matriz de confusion

Guardada en: `matriz_confusion.png`

### Accuracy correcta/incorrecta (binaria)

- **75.98%** -- ¿el modelo distingue bien si la palabra se dijo bien o mal, más allá de si acierta el subtipo exacto de error?

### Confianza del modelo (umbral 40%)

- Predicciones con confianza suficiente para mostrarle un resultado al chico: **2671/3064** (87.17%)
- Sin confianza suficiente (el server pediría repetir la toma): **393** (12.83%)

### Latencia de inferencia

- Promedio: **17.18 ms** (batch=1, cuda)
