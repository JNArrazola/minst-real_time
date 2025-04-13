"""  
main.py
Clase principal de la aplicación.
"""

import sys
import numpy as np
import tensorflow as tf
from PyQt6.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QPushButton
from PyQt6.QtGui import QPixmap, QPainter, QPen, QColor, QImage
from PyQt6.QtCore import Qt, QPoint, QTimer
from scipy.ndimage import center_of_mass, shift
from src.visualizer import NeuralNetworkVisualizer

"""  
Clase para crear un lienzo de dibujo donde el usuario puede dibujar un número.
El lienzo se utiliza para capturar la entrada del usuario y predecir el número dibujado.
El lienzo se implementa utilizando QPixmap y permite al usuario dibujar con el mouse.
La clase DrawingCanvas maneja eventos de mouse para permitir el dibujo y la limpieza del lienzo.  
La clase también incluye un temporizador para activar la predicción después de un breve período de inactividad.
"""
class DrawingCanvas(QWidget):

    """  
    Constructor de la clase DrawingCanvas.
    """
    def __init__(self):
        super().__init__()
        self.setFixedSize(280, 280)
        self.canvas = QPixmap(self.size())
        self.canvas.fill(Qt.GlobalColor.white)
        self.last_point = QPoint()
        self.drawing = False

        self.timer = QTimer(self) # Temporizador para activar la predicción
        self.timer.setInterval(200) # # Intervalo de 200 ms (si no ponía temporizador, el programa crasheaba masivamente - Joshua)
        self.timer.timeout.connect(self.trigger_prediction) 

    """  
    Método para manejar el evento de clic del mouse.
    Cuando el usuario hace clic con el botón izquierdo del mouse, se inicia el dibujo y se guarda la posición inicial.
    """
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.last_point = event.position().toPoint()
            self.drawing = True
            self.timer.start()

    """
    Método para manejar el evento de movimiento del mouse.
    Cuando el usuario mueve el mouse mientras dibuja, se dibuja una línea desde la última posición hasta la nueva posición.
    """
    def mouseMoveEvent(self, event):
        if self.drawing:
            painter = QPainter(self.canvas)
            pen = QPen(Qt.GlobalColor.black, 18, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
            painter.setPen(pen)
            painter.drawLine(self.last_point, event.position().toPoint())
            self.last_point = event.position().toPoint() # Actualiza la posición del último punto
            self.update() # Actualiza el lienzo para mostrar el dibujo

    """
    Método para manejar el evento de liberación del mouse.
    Cuando el usuario suelta el botón izquierdo del mouse, se detiene el dibujo y se activa la predicción.
    """
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.drawing = False
            self.timer.stop()
            self.trigger_prediction()

    """
    Método para manejar el evento de pintura del lienzo.
    Se utiliza un QPainter para dibujar el contenido del lienzo en la ventana.
    """
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.drawPixmap(0, 0, self.canvas)

    """
    Método para limpiar el lienzo.
    Se llena el lienzo con color blanco y actualiza la vista.
    """
    def clear(self):
        self.canvas.fill(Qt.GlobalColor.white)
        self.update()

    """
    Método para obtener la imagen del lienzo.
    Convierte el lienzo a un formato de imagen en escala de grises y lo convierte a un arreglo numpy.
    El arreglo se normaliza dividiendo por 255.0.
    """
    def get_image(self):
        image = self.canvas.toImage().convertToFormat(QImage.Format.Format_Grayscale8)
        ptr = image.bits()
        ptr.setsize(image.width() * image.height())
        arr = np.frombuffer(ptr, np.uint8).reshape((280, 280))
        arr = arr / 255.0
        return arr

    """
    Método para activar la predicción.
    Si el temporizador está activo, se detiene y se inicia nuevamente.
    Si el temporizador no está activo, se inicia la predicción.
    """
    def trigger_prediction(self):
        if self.parent():
            self.parent().predict_digit()

"""
Clase principal de la aplicación.
La clase MainWindow crea la ventana principal de la aplicación y contiene el lienzo de dibujo y el botón de limpieza.
La clase también carga el modelo de TensorFlow y maneja la predicción del número dibujado.
La clase MainWindow utiliza la clase DrawingCanvas para permitir al usuario dibujar un número y predecirlo utilizando el modelo cargado.
"""
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Reconocedor de Números MNIST - Visualización Dinámica")

        self.canvas = DrawingCanvas()
        self.label = QLabel("Dibuja un número")
        self.label.setStyleSheet("font-size: 20px;")

        self.clear_button = QPushButton("Limpiar")
        self.clear_button.clicked.connect(self.canvas.clear)

        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        layout.addWidget(self.label)
        layout.addWidget(self.clear_button)
        self.setLayout(layout)

        self.model = tf.keras.models.load_model("model.h5")
        self.visualizer = NeuralNetworkVisualizer()

        self.weights_input_hidden = self.model.layers[1].get_weights()[0]  # (784, 128)
        self.weights_hidden_output = self.model.layers[2].get_weights()[0]  # (128, 10)

    """  
    Método para predecir el dígito dibujado en el lienzo.
    Se obtiene la imagen del lienzo, se redimensiona y se normaliza.
    Luego, se aplica un desplazamiento para centrar la imagen y se realiza la predicción utilizando el modelo cargado.
    Se calcula la activación de la capa oculta y se aplica softmax a la salida.
    Finalmente, se muestra la predicción y la confianza en la etiqueta.

    El método también actualiza la visualización de la red neuronal con la imagen, la activación y los pesos.
    Se utiliza la función center_of_mass para centrar la imagen y la función shift para desplazarla.
    """
    def predict_digit(self):
        raw = self.canvas.get_image()
        resized = tf.image.resize(raw[..., np.newaxis], (28, 28)).numpy().squeeze()

        img = 1.0 - resized
        img[img < 0.2] = 0.0

        # Centrar la imagen
        cy, cx = center_of_mass(img)
        shift_y = int(14 - cy)
        shift_x = int(14 - cx)
        img = shift(img, [shift_y, shift_x], mode='constant')

        img_reshaped = img.reshape(1, 28, 28, 1) 
        flat_input = img.reshape(1, 784)

        hidden_raw = flat_input @ self.weights_input_hidden
        hidden_act = tf.nn.relu(hidden_raw).numpy()[0]

        output_raw = hidden_act @ self.weights_hidden_output
        output_softmax = tf.nn.softmax(output_raw).numpy()

        predicted = np.argmax(output_softmax)
        confidence = output_softmax[predicted]
        self.label.setText(f"Parece un: {predicted} ({confidence * 100:.1f}%)")

        # Actualizar la visualización de la red neuronal
        self.visualizer.update_view(
            img,
            hidden_act,
            output_softmax,
            self.weights_input_hidden,
            self.weights_hidden_output
        )

"""  
Método principal para ejecutar la aplicación.
"""
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
