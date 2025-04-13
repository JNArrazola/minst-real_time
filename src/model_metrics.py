"""  
Archivo para evaluar el modelo entrenado en el conjunto de datos MNIST.
Este archivo carga el modelo entrenado, evalúa su rendimiento en el conjunto de prueba
y genera un reporte de clasificación y una matriz de confusión.

Este archivo no se ejecuta directamente, sino que se utiliza como un módulo para evaluar el modelo.
"""

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Cargar modelo y datos
model = tf.keras.models.load_model("model.h5")
(_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalizar imágenes
x_test = x_test / 255.0
x_test = x_test[..., np.newaxis]  # (n, 28, 28, 1)

# Convertir etiquetas a one-hot encoding para evaluación
y_test_categorical = to_categorical(y_test, num_classes=10)

# Evaluación del modelo
loss, acc = model.evaluate(x_test, y_test_categorical, verbose=0)
print(f"Accuracy del modelo en test: {acc:.4f}")

# Obtener predicciones
y_pred_probs = model.predict(x_test, verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)

# Reporte de clasificación
print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred, digits=4))

# Matriz de confusión
cm = confusion_matrix(y_test, y_pred)

# Graficar matriz de confusión
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel("Etiqueta predicha")
plt.ylabel("Etiqueta verdadera")
plt.title("Matriz de confusión - MNIST")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300)
plt.show()
