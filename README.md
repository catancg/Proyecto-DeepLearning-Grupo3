# TP de Redes Neuronales: clasificación de movimiento con una CNN temporal

Clasificación de la etiqueta publicada **correcta / incorrecta** de un único gesto (gesto 2 de IntelliRehabDS), con evaluación por participante y comparación contra regresión logística.

**Autores:** Laura Daniela Muñoz Torres, Anna Carla, Carlos Catán, Jonathan Caceres Prada.

**Pregunta:** ¿qué desempeño tiene una CNN temporal pequeña para reproducir la etiqueta publicada correcta/incorrecta del gesto 2, frente a una regresión logística sobre características de movimiento, evaluando siempre en participantes que el modelo no vio al entrenar?

- **Unidad de análisis:** una secuencia completa (una repetición).
- **Agrupación:** participante.
- El diseño quedó fijado antes de entrenar en `criterios_v1.md`.
- Este trabajo reproduce una etiqueta publicada del dataset. No es validación clínica.

Fuente: [IntelliRehabDS v2.0.1](https://zenodo.org/records/4610859) (Miron et al., CC BY 4.0).

## Cómo ejecutarlo

El notebook está pensado para **Google Colab**, en **CPU** (no requiere GPU).

1. Subir `TP_Redes_DL.ipynb` a Colab y conectar un entorno de ejecución Python 3.
2. Ejecutar las celdas en orden. La celda de carga pide **SkeletonData.zip** (aprox. 199 MB).
3. El ZIP se verifica por MD5 (`c243bcdbd1492e4928032e158accdd29`) y por la presencia de archivos `.txt` bajo `Simplified/`.
4. Al terminar, la última celda descarga `resultados_TP_IRDS.zip`. Guardar también el notebook con sus salidas.

El directorio de trabajo en Colab es `/content/irds_tp`. Si el ZIP canónico no está, el código puede usar `simplified_partial.zip` (reempaquetado de `Simplified`, verificado contra `audit_manifest.json`).

Esta copia corrige la normalización (estadísticas calculadas por modelo interno), exporta métricas no estimables como nulo y deja las predicciones para auditoría. Las conclusiones hay que redactarlas a partir de las salidas de **esta** ejecución.

No publicar identificadores individuales del dataset.

## Dependencias

`numpy`, `scikit-learn`, `torch`, `matplotlib`. La primera celda de código las instala si faltan. En Colab se usa además `google.colab.files` para subir el ZIP y descargar los resultados.

## Datos y exclusiones

Cada archivo se llama `Sujeto_Fecha_Gesto_Repetición_Etiqueta_Postura.txt`. El identificador del sujeto solo agrupa las particiones: no entra al modelo. Tampoco entran el grupo ni la postura.

Flujo de exclusiones, en este orden:

1. Quedarse con el **gesto 2**.
2. Quitar al sujeto **217** (grabación interrumpida, aviso de Zenodo).
3. Quitar la **etiqueta 3** (ambigua). Quedan las etiquetas 1 y 2.

La clase de interés es la etiqueta **2 = incorrecta**. Los sujetos `2xx` se tratan como pacientes según el artículo, y esa marca se usa solo para auditar.

En el gesto 2, todas las secuencias incorrectas son de pacientes y todos los controles son correctos. Por eso el análisis B (solo pacientes) controla el sesgo de reconocer “quién es paciente” sin mirar la calidad del movimiento.

## Entradas

Normalización **por secuencia** (no aprende de otros datos): cada fotograma se centra en la base de la columna y se divide por la longitud del tronco de esa persona. Reduce diferencias de posición y de escala corporal.

| Entrada | Uso |
| --- | --- |
| Secuencia remuestreada a 64 fotogramas, 75 canales (25 articulaciones × 3) | CNN |
| `log(número de fotogramas)` | Indicador aproximado de duración (el remuestreo borra la longitud original). Se concatena antes de la salida de la CNN. |
| 15 características de ángulos | Regresión logística. Ambos brazos (hombro–codo–muñeca), más inclinación del tronco y el log de la duración. |
| Primer fotograma ya normalizado (sin movimiento) | Modelo estático de control |

La estandarización de la CNN (media y desvío de la secuencia y de la duración) se calcula **solo con el entrenamiento** de cada modelo interno.

## Modelos

Tres modelos, todos con validación anidada:

1. **Estático.** Regresión logística sobre el primer fotograma. Explora si la postura inicial ya predice la etiqueta. No identifica el atajo que pueda estar usando la CNN.
2. **Logística.** Regresión logística sobre las 15 características, con `StandardScaler`, `class_weight="balanced"` y `C` elegido en `{0.01, 0.1, 1, 10}` por AUC interna.
3. **CNN temporal.** 22.594 parámetros entrenables.
   - Conv1d 75→32, kernel 5, padding 2 → BatchNorm → ReLU → MaxPool (tiempo a la mitad).
   - Conv1d 32→64, kernel 5, padding 2 → BatchNorm → ReLU → promedio global.
   - Dropout 0,30. Se concatena la duración y una capa lineal de 65→1.
   - Pérdida: entropía cruzada binaria con logits y peso mayor para la clase minoritaria, calculado solo en entrenamiento.
   - Adam, tasa de aprendizaje 0,001, weight decay 0,0001, lotes de hasta 32 secuencias, hasta 60 épocas.
   - La sigmoide convierte la salida en una puntuación. Esa puntuación no está calibrada para uso clínico.

## Validación

Hay pocos participantes con secuencias incorrectas (el notebook parte de 10), así que un único test reservado sería inestable. El esquema es validación cruzada **anidada y agrupada por participante**:

- **Bucle externo:** estima el desempeño. Análisis **A** (todos): 5 particiones. Análisis **B** (solo pacientes): 4 particiones.
- **Bucle interno:** 3 particiones, solo con el entrenamiento externo. Elige el hiperparámetro (`C` o número de épocas) y el **umbral de Youden**.
- Semillas **0, 1 y 2**. Todas las repeticiones de una persona permanecen juntas. Las semillas no son participantes nuevos ni estudios independientes.
- El fold externo de prueba no elige épocas ni umbral.

**Épocas de la CNN.** El bucle interno recorre las 60 épocas y guarda el estado de cada una. Se elige la época de menor pérdida media de validación interna y se reutilizan esos estados: la predicción de prueba es el promedio de los tres modelos internos, cada uno con su propia normalización. No es una detención anticipada, y no se reentrena sobre todo el entrenamiento externo.

El umbral sale de predicciones internas fuera de muestra; la prueba usa el promedio de modelos. Esa diferencia queda documentada como limitación.

## Métricas

Sobre las predicciones agregadas de cada semilla: sensibilidad, especificidad, precisión, F1, exactitud balanceada y AUC (media ± desvío entre semillas), más la exactitud balanceada entre folds.

Complementarias: accuracy, cantidad de errores, MCC, PR-AUC, Brier y log-loss. La comparación es **CNN menos logística** sobre las mismas predicciones fuera de muestra.

Cuando un fold tiene una sola clase, sensibilidad, especificidad y exactitud balanceada de la clase ausente no se estiman: se exportan como nulo y se excluyen del promedio por fold, informando cuántos folds quedaron fuera. Las métricas agrupadas por semilla usan todas las predicciones externas de esa semilla.

La incertidumbre se estima con **bootstrap por participante** (2000 remuestreos, semilla 0) de sensibilidad, especificidad y exactitud balanceada. Se remuestrean personas, no secuencias.

Las pérdidas de entrenamiento (ponderada) y de validación interna (no ponderada) no se restan: no son una brecha de generalización. En las curvas, azul es entrenamiento, naranja es validación interna y la línea vertical es la época elegida.

## Salidas

Todo queda en `resultados/`:

| Archivo | Contenido |
| --- | --- |
| `curvas_cnn.png` | Curvas por fold externo, semilla 0, análisis A y B |
| `resultados.json` | Entorno, exclusiones, métricas, intervalos y desempeño por participante |
| `particiones.json` | Sujetos de prueba, tamaños y cantidad de incorrectas por fold |
| `predicciones_auditoria.json` | Predicciones y probabilidades fuera de muestra, para auditar sin reentrenar |
| `cnn_demo_todos_los_datos.pt` | Pesos de demostración entrenados con todos los datos. No tiene evaluación independiente |

`resultados_TP_IRDS.zip` empaqueta esa carpeta. No incluye el dataset.

## Lectura de los resultados

Redactar las conclusiones solo con las salidas de la ejecución corregida. El modelo estático y el análisis solo-pacientes sirven para ver si el desempeño se sostiene cuando se quita la pista de “control vs. paciente” y cuando se mira la postura inicial sin movimiento. La especificidad sobre secuencias correctas de pacientes (análisis A) se imprime aparte por el mismo motivo.
