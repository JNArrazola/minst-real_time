"""  
src/visualizer.py
Clase para visualizar la red neuronal.
"""

import numpy as np
import matplotlib.pyplot as plt
from PyQt6.QtWidgets import QWidget, QVBoxLayout
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.patches as patches

"""  
Clase NeuralNetworkVisualizer
Esta clase se encarga de visualizar la red neuronal simplificada.
La visualización incluye la estructura de la red neuronal, los nodos de entrada, ocultos y salida,
así como las conexiones entre ellos.

La clase utiliza Matplotlib para crear la visualización y PyQt6 para la interfaz gráfica.
"""
class NeuralNetworkVisualizer(QWidget):

    """  
    Constructor de la clase NeuralNetworkVisualizer.
    Inicializa la ventana, el lienzo y la estructura de la red neuronal.
    Se definen los nodos de entrada, ocultos y salida, así como sus posiciones.
    Se llama al método _draw_static_structure para dibujar la estructura estática de la red neuronal.
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Red Neuronal Simplificada")
        self.setGeometry(100, 100, 800, 500)

        self.canvas = FigureCanvas(Figure(figsize=(8, 5)))
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)

        self.ax = self.canvas.figure.subplots()
        self.ax.axis('off')

        # Definición de la red neuronal
        # Esto es totalmente arbitrario, pero estos valores funcionan bien en la visualización, 
        # sobretodo para la red neuronal de MNIST y evitar el lag
        self.input_nodes = 10
        self.hidden_nodes = 16
        self.output_nodes = 10

        self.input_circles = []
        self.hidden_circles = []
        self.output_circles = []

        self._draw_static_structure()
        self.show()

    """
    Método para dibujar la estructura estática de la red neuronal.
    Se definen las posiciones de los nodos de entrada, ocultos y salida.
    Se dibujan las conexiones entre los nodos y se añaden los círculos que representan los nodos.
    Se utilizan diferentes colores para los nodos de entrada, ocultos y salida.
    * Las posiciones de los nodos se definen mediante la función positions, que calcula las coordenadas de los nodos en función del número de nodos y su posición en la red.
    * Las conexiones entre los nodos se dibujan utilizando líneas de color gris.
    * Los nodos se representan como círculos de diferentes colores utilizando la función _add_circle.
    """
    def _draw_static_structure(self):
        self.ax.clear()
        self.ax.axis('off')

        # Posiciones fijas
        def positions(n, x, y_start, y_end):
            return [(x, y) for y in np.linspace(y_start, y_end, n)]

        self.input_pos = positions(self.input_nodes, 0.1, 0.1, 0.9)
        self.hidden_pos = positions(self.hidden_nodes, 0.5, 0.1, 0.9)
        self.output_pos = positions(self.output_nodes, 0.9, 0.1, 0.9)

        # Conexiones input -> hidden
        for (x1, y1) in self.input_pos:
            for (x2, y2) in self.hidden_pos:
                self.ax.plot([x1, x2], [y1, y2], color='gray', linewidth=0.3, alpha=0.4)

        # Conexiones hidden -> output
        for (x1, y1) in self.hidden_pos:
            for (x2, y2) in self.output_pos:
                self.ax.plot([x1, x2], [y1, y2], color='gray', linewidth=0.4, alpha=0.4)

        # Nodos (círculos)
        self.input_circles = [self._add_circle(x, y, 0.015, 'Greys', 0.0) for (x, y) in self.input_pos]
        self.hidden_circles = [self._add_circle(x, y, 0.02, 'viridis', 0.0) for (x, y) in self.hidden_pos]
        self.output_circles = [self._add_circle(x, y, 0.03, 'plasma', 0.0) for (x, y) in self.output_pos]

        # Etiquetas en nodos de salida
        for idx, (x, y) in enumerate(self.output_pos):
            self.ax.text(x + 0.02, y, f"{idx}", va='center', fontsize=8)

        self.canvas.draw()

    """
    Método para añadir un círculo en la visualización.
    Se utiliza para representar los nodos de entrada, ocultos y salida.
    Se define el color del círculo utilizando un mapa de colores (cmap).
    * Se utiliza la función patches.Circle para crear el círculo y se añade al eje de la visualización.
    * Se devuelve el círculo creado para poder actualizar su color posteriormente.
    """
    def _add_circle(self, x, y, r, cmap_name, value):
        cmap = plt.get_cmap(cmap_name)
        circle = patches.Circle((x, y), r, color=cmap(value))
        self.ax.add_patch(circle)
        return circle

    """
    Método para actualizar la visualización de la red neuronal.
    Se actualizan los colores de los nodos de entrada, ocultos y salida en función de las activaciones.
    * Se utiliza la función np.interp para interpolar los valores de entrada y normalizar las activaciones de las capas ocultas y de salida.
    * Se actualizan los colores de los círculos utilizando el mapa de colores correspondiente.
    * Se llama a la función draw_idle para actualizar la visualización de manera eficiente.
    """
    def update_view(self, input_image, hidden_activations, output_activations, *_):
        # INPUT: usar promedio vertical (28 -> 10)
        input_image = input_image.reshape(28, 28)
        avg_rows = np.mean(input_image, axis=1)
        input_values = np.interp(np.linspace(0, 27, self.input_nodes), np.arange(28), avg_rows)

        # HIDDEN y OUTPUT: normalizar
        hidden_values = hidden_activations[:self.hidden_nodes]
        hidden_values = hidden_values / np.max(hidden_values) if np.max(hidden_values) != 0 else hidden_values

        output_values = output_activations[:self.output_nodes]
        output_values = output_values / np.max(output_values) if np.max(output_values) != 0 else output_values

        # Actualizar colores
        for circle, val in zip(self.input_circles, input_values):
            circle.set_color(plt.cm.Greys(val))

        for circle, val in zip(self.hidden_circles, hidden_values):
            circle.set_color(plt.cm.viridis(val))

        for circle, val in zip(self.output_circles, output_values):
            circle.set_color(plt.cm.plasma(val))

        self.canvas.draw_idle()
