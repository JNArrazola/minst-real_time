import sys
import numpy as np
import tensorflow as tf
from PyQt6.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QPushButton
from PyQt6.QtGui import QPixmap, QPainter, QPen, QColor, QImage
from PyQt6.QtCore import Qt, QPoint
from scipy.ndimage import center_of_mass, shift

class DrawingCanvas(QWidget):
    def __init__(self):
        super().__init__()
        self.setFixedSize(280, 280)  
        self.canvas = QPixmap(self.size())
        self.canvas.fill(Qt.GlobalColor.white)
        self.last_point = QPoint()
        self.drawing = False

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.last_point = event.position().toPoint()
            self.drawing = True

    def mouseMoveEvent(self, event):
        if self.drawing:
            painter = QPainter(self.canvas)
            pen = QPen(Qt.GlobalColor.black, 18, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
            painter.setPen(pen)
            painter.drawLine(self.last_point, event.position().toPoint())
            self.last_point = event.position().toPoint()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.drawing = False
            self.parent().predict_digit()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.drawPixmap(0, 0, self.canvas)

    def clear(self):
        self.canvas.fill(Qt.GlobalColor.white)
        self.update()

    def get_image(self):
        image = self.canvas.toImage().convertToFormat(QImage.Format.Format_Grayscale8)
        ptr = image.bits()
        ptr.setsize(image.width() * image.height())
        arr = np.frombuffer(ptr, np.uint8).reshape((280, 280))
        arr = arr / 255.0  
        return arr

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Reconocedor de Números MNIST - Precisión Mejorada")

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

    def predict_digit(self):
        raw = self.canvas.get_image()
        resized = tf.image.resize(raw[..., np.newaxis], (28, 28)).numpy().squeeze()

        img = 1.0 - resized

        img[img < 0.2] = 0.0

        cy, cx = center_of_mass(img)
        shift_y = int(14 - cy)
        shift_x = int(14 - cx)
        img = shift(img, [shift_y, shift_x], mode='constant')

        img = img.reshape(1, 28, 28, 1)

        prediction = self.model.predict(img, verbose=0)[0]
        digit = np.argmax(prediction)
        confidence = prediction[digit]
        self.label.setText(f"Parece un: {digit} ({confidence * 100:.1f}%)")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
