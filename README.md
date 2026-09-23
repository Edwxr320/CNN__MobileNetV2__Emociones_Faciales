# Reconocimiento de Emociones Faciales con CNN y MobileNetV2

Proyecto de clasificación de emociones faciales mediante Deep Learning, desarrollado con TensorFlow y Keras.

El proyecto implementa y compara dos enfoques de clasificación de imágenes:

1. Una red neuronal convolucional (CNN) construida desde cero.
2. Un modelo MobileNetV2 mediante Transfer Learning y Fine-Tuning.

El sistema clasifica imágenes faciales en siete categorías emocionales y permite realizar predicciones sobre imágenes propias.

---

## Descripción

El objetivo del proyecto es desarrollar un modelo capaz de clasificar imágenes de rostros según una de las siguientes emociones:

* Angry
* Disgust
* Fear
* Happy
* Neutral
* Sad
* Surprise

Para ello, se realiza un flujo completo de trabajo de Deep Learning:

```text
Carga del dataset
      ↓
Exploración de datos
      ↓
Preprocesamiento
      ↓
Data Augmentation
      ↓
Balanceo mediante class weights
      ↓
Entrenamiento CNN
      ↓
Evaluación
      ↓
Transfer Learning con MobileNetV2
      ↓
Comparación de modelos
      ↓
Predicción en nuevas imágenes
```

---

## Dataset

El notebook utiliza un dataset organizado en directorios por clase.

Se trabajó con un total de:

**49,779 imágenes**

Distribuidas entre siete clases emocionales.

La partición utilizada fue:

```text
Entrenamiento: 39,824 imágenes (80%)
Validación:     9,955 imágenes (20%)
```

La clase con mayor cantidad de imágenes fue `happy`, con **11,398 imágenes**, mientras que `angry` fue la clase con menor cantidad, con **5,920 imágenes**.

Debido al desbalance entre clases, se utilizaron pesos de clase (`class_weight`) durante el entrenamiento.

> El notebook no documenta en detalle el origen o nombre oficial del dataset, por lo que esta descripción se limita a la estructura y cantidades observadas directamente en el proyecto.

---

## Preprocesamiento

Para la CNN construida desde cero se utilizaron imágenes de:

```text
48 × 48 píxeles
Escala de grises
1 canal
```

Los valores de los píxeles se normalizan al rango:

```text
[0, 1]
```

mediante una escala de `1/255`.

### Data Augmentation

Durante el entrenamiento se aplicaron transformaciones para aumentar la variabilidad de las imágenes:

* Rotación de hasta ±15°
* Desplazamiento horizontal
* Desplazamiento vertical
* Volteo horizontal
* Zoom
* Shear
* Relleno mediante `nearest`

El conjunto de validación no utiliza Data Augmentation; únicamente se normaliza.

---

# Modelo 1 — CNN desde cero

Se desarrolló una red convolucional de cuatro bloques.

Cada bloque utiliza una combinación de:

```text
Conv2D
BatchNormalization
Conv2D
BatchNormalization
MaxPooling2D
Dropout
```

La arquitectura aumenta progresivamente el número de filtros:

```text
32 → 64 → 128 → 256
```

Posteriormente se utilizan:

```text
GlobalAveragePooling2D
Dense(512)
BatchNormalization
Dropout(0.5)
Dense(256)
Dropout(0.3)
Dense(7, Softmax)
```

### Parámetros

El modelo CNN tiene:

```text
1,442,279 parámetros
```

La salida final contiene siete probabilidades, una por cada emoción.

---

## Entrenamiento CNN

Se utilizaron:

```text
Optimizador: Adam
Learning Rate inicial: 0.001
Batch Size: 64
Épocas máximas: 40
EarlyStopping
ReduceLROnPlateau
ModelCheckpoint
Class Weight
Seed: 42
```

El mejor modelo se guarda como:

```text
best_emotion_cnn.h5
```

### Resultado obtenido

El entrenamiento registrado en el notebook alcanzó:

```text
Mejor val_accuracy: 59.75%
Mejor época:        17
Accuracy global:    59.75%
```

En la época final registrada:

```text
Train accuracy: 74.83%
Validation accuracy: 59.02%
```

Esto muestra una diferencia entre entrenamiento y validación, por lo que el modelo presenta cierta brecha de generalización.

---

# Evaluación del modelo CNN

El notebook genera una evaluación utilizando:

* Accuracy global
* Precision
* Recall
* F1-score
* Matriz de confusión
* Matriz de confusión normalizada
* Análisis de errores entre clases

Resultado global registrado:

```text
Accuracy: 59.75%
```

También se genera un análisis de las confusiones más frecuentes entre emociones.

Los archivos gráficos generados incluyen:

```text
curvas_aprendizaje.png
matriz_confusion.png
predicciones_ejemplo.png
```

---

# Modelo 2 — Transfer Learning con MobileNetV2

Como segundo enfoque se implementa MobileNetV2 utilizando pesos preentrenados de ImageNet.

Para este modelo se utilizan imágenes de:

```text
96 × 96 píxeles
RGB
3 canales
```

Aunque las imágenes originales son de escala de grises, se utilizan tres canales para cumplir con el formato de entrada requerido por MobileNetV2.

## Arquitectura

MobileNetV2 se utiliza como extractor de características sin su cabeza clasificadora original.

Después se agrega una cabeza personalizada:

```text
MobileNetV2
      ↓
GlobalAveragePooling2D
      ↓
Dense(512)
      ↓
BatchNormalization
      ↓
Dropout(0.5)
      ↓
Dense(256)
      ↓
Dropout(0.3)
      ↓
Dense(7, Softmax)
```

El modelo cuenta con:

```text
Parámetros totales:       3,049,031
Parámetros entrenables:     790,023
Parámetros congelados:    2,259,008
```

---

## Fase 1 — Transfer Learning

Primero se congelan las capas de MobileNetV2 y se entrena únicamente la cabeza de clasificación.

Configuración principal:

```text
Learning Rate: 0.001
Batch Size: 64
Épocas máximas: 15
EarlyStopping
Class Weight
```

Resultado:

```text
Mejor val_accuracy: 43.48%
```

---

## Fase 2 — Fine-Tuning

Posteriormente se descongelan las capas superiores de MobileNetV2.

Se mantienen congeladas las primeras 100 capas y se entrenan las capas superiores con un learning rate reducido:

```text
Learning Rate: 1e-5
```

Durante el proceso también se utilizan:

* EarlyStopping
* ReduceLROnPlateau
* ModelCheckpoint
* Class Weight

El mejor modelo se guarda como:

```text
best_emotion_mobilenet.h5
```

### Resultado obtenido

El mejor valor registrado fue:

```text
MobileNetV2 val_accuracy: 50.27%
```

---

# Comparación de modelos

Resultados registrados directamente en el notebook:

| Modelo                  | Mejor Validation Accuracy |
| ----------------------- | ------------------------: |
| CNN desde cero          |                    59.75% |
| MobileNetV2 Fine-Tuning |                    50.27% |

En este entrenamiento concreto, la CNN desarrollada desde cero obtuvo una mayor `validation accuracy` que la implementación de MobileNetV2.

Este resultado corresponde a la configuración, dataset y ejecución registrados en este notebook y no implica que una arquitectura sea universalmente superior a la otra.

---

# Predicción de emociones

El proyecto incluye funciones para realizar predicciones sobre nuevas imágenes.

El proceso consiste en:

```text
Cargar imagen
      ↓
Redimensionar a 48 × 48
      ↓
Convertir a escala de grises
      ↓
Normalizar píxeles
      ↓
Ejecutar modelo CNN
      ↓
Obtener probabilidades
      ↓
Seleccionar clase con mayor probabilidad
```

La función de predicción devuelve:

* emoción detectada
* confianza
* probabilidades de las siete clases

También se genera un gráfico con las probabilidades correspondientes a cada emoción.

---

# Predicción con imágenes propias

El notebook incluye una sección adicional que permite subir fotografías desde Google Colab y obtener una predicción.

El usuario puede cargar imágenes en formatos como:

```text
JPG
JPEG
PNG
```

Las imágenes se transforman automáticamente a:

```text
48 × 48
Escala de grises
Normalización [0,1]
```

El sistema muestra:

* imagen cargada
* emoción predicha
* nivel de confianza
* distribución de probabilidades por emoción

> La predicción es una clasificación experimental del modelo y no constituye una evaluación psicológica, clínica ni una determinación objetiva del estado emocional de una persona.

---

# Archivos generados

Durante la ejecución del notebook se generan o guardan archivos como:

```text
best_emotion_cnn.h5
emotion_cnn_final.h5

best_emotion_mobilenet.h5
emotion_mobilenet_final.h5

class_names.json
metricas_finales.txt

curvas_aprendizaje.png
matriz_confusion.png
predicciones_ejemplo.png
```

El archivo:

```text
class_names.json
```

mantiene la correspondencia entre índice y emoción:

```text
0 → angry
1 → disgust
2 → fear
3 → happy
4 → neutral
5 → sad
6 → surprise
```

---

# Tecnologías utilizadas

* Python
* TensorFlow
* Keras
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Google Colab

---

# Requisitos

El notebook instala las siguientes librerías:

```bash
pip install matplotlib
pip install seaborn
pip install scikit-learn
pip install tensorflow
```

Se recomienda utilizar Google Colab para ejecutar el proyecto, especialmente para aprovechar aceleración GPU durante el entrenamiento.

El notebook fue configurado para utilizar una GPU NVIDIA T4 cuando está disponible.

---

# Cómo ejecutar

## 1. Abrir el notebook

Abrir:

```text
CNN__MobileNetV2__Emociones_Faciales.ipynb
```

en Google Colab.

## 2. Ejecutar la instalación

Ejecutar la primera celda para instalar las dependencias.

## 3. Cargar el dataset

El notebook solicita subir el archivo ZIP del dataset.

La estructura esperada contiene una carpeta:

```text
processed_data/
```

con una subcarpeta por emoción.

Ejemplo:

```text
processed_data/
├── angry/
├── disgust/
├── fear/
├── happy/
├── neutral/
├── sad/
└── surprise/
```

## 4. Ejecutar las celdas en orden

El notebook está organizado en 12 etapas:

```text
1. Instalación de librerías
2. Carga y descompresión del dataset
3. Exploración visual y distribución
4. Preprocesamiento y Data Augmentation
5. Arquitectura CNN
6. Entrenamiento
7. Curvas de aprendizaje
8. Evaluación y matriz de confusión
9. Transfer Learning con MobileNetV2
10. Predicciones visuales
11. Guardado y descarga de modelos
12. Predicción con imágenes propias
```

---

# Estructura del notebook

```text
CNN__MobileNetV2__Emociones_Faciales.ipynb
│
├── Preparación del entorno
├── Carga del dataset
├── Exploración
├── Preprocesamiento
├── CNN desde cero
├── Entrenamiento
├── Evaluación
├── Transfer Learning
├── Fine-Tuning
├── Comparación de modelos
├── Predicciones
└── Exportación de modelos
```

---

# Limitaciones

Este proyecto corresponde a un experimento de clasificación de emociones mediante imágenes y presenta algunas limitaciones:

* La precisión depende de las características y distribución del dataset utilizado.
* Existe desbalance entre las clases.
* La validación utilizada corresponde al 20% del dataset mediante `validation_split`.
* El notebook no implementa detección facial previa ni recorte automático del rostro; la imagen cargada se procesa directamente para la clasificación.
* Las imágenes propias pueden presentar resultados variables debido a iluminación, posición del rostro, calidad de imagen y diferencias respecto al dataset de entrenamiento.
* La confianza mostrada corresponde a la probabilidad producida por el modelo y no debe interpretarse como certeza.
* El seguimiento de una emoción en una persona real requiere considerar factores contextuales y no puede determinarse únicamente a partir de una imagen.

---

# Resultados destacados

```text
Dataset:
49,779 imágenes

Clases:
7 emociones

Train:
39,824 imágenes

Validation:
9,955 imágenes

CNN:
59.75% mejor validation accuracy

MobileNetV2 + Fine-Tuning:
50.27% mejor validation accuracy
```

---

# Modelos disponibles

### CNN

Modelo convolucional desarrollado desde cero para clasificación de siete emociones.

```text
emotion_cnn_final.h5
```

### MobileNetV2

Modelo basado en Transfer Learning y Fine-Tuning.

```text
emotion_mobilenet_final.h5
```

---

# Propósito del proyecto

Proyecto académico orientado al aprendizaje práctico de:

* Redes Neuronales Convolucionales
* Clasificación de imágenes
* Deep Learning
* Transfer Learning
* Fine-Tuning
* Data Augmentation
* Balanceo de clases
* Evaluación de modelos
* Visualización de resultados
* Inferencia sobre nuevas imágenes

---

## Autor

**Edwar Alama**

Estudiante de Ingeniería de Software con Inteligencia Artificial.

GitHub: [Edwxr320](https://github.com/Edwxr320)

LinkedIn: [Edwar Alama](https://www.linkedin.com/in/edwar-alama)
