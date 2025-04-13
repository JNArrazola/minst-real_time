import numpy as np
from PyQt6.QtWidgets import QWidget, QVBoxLayout
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.patches as patches
import matplotlib.pyplot as plt  

class NeuralNetworkVisualizer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Visualización de Red Neuronal")
        self.setGeometry(100, 100, 900, 600)

        self.canvas = FigureCanvas(Figure(figsize=(9, 6)))
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)

        self.ax = self.canvas.figure.subplots()
        self.show()

    def update_view(self, input_image, hidden_activations, output_activations, w_ih=None, w_ho=None):
        self.ax.clear()
        self.ax.axis('off')

        input_nodes = 20
        hidden_nodes = 32
        output_nodes = 10

        input_image = input_image.reshape(28, 28)

        
        input_image = input_image.reshape(28, 28)
        input_image_resized = np.mean(input_image, axis=1)
        input_activations = np.interp(np.linspace(0, 27, input_nodes), np.arange(28), input_image_resized)
        
        hidden_activations = hidden_activations[:hidden_nodes]
        output_activations = output_activations[:output_nodes]

        def node_positions(n, x, y_start, y_end):
            return [(x, y) for y in np.linspace(y_start, y_end, n)]

        input_pos = node_positions(input_nodes, 0.1, 0.1, 0.9)
        hidden_pos = node_positions(hidden_nodes, 0.5, 0.05, 0.95)
        output_pos = node_positions(output_nodes, 0.9, 0.1, 0.9)

        # Conexiones input → hidden (si hay pesos)
        if w_ih is not None:
            for i, (x1, y1) in enumerate(input_pos):
                for j, (x2, y2) in enumerate(hidden_pos):
                    w = w_ih[i % w_ih.shape[0], j]
                    color = 'red' if w < 0 else 'blue'
                    alpha = min(1.0, abs(w) / np.max(np.abs(w_ih)))
                    self.ax.plot([x1, x2], [y1, y2], color=color, alpha=alpha, linewidth=0.3)

        # Conexiones hidden → output
        if w_ho is not None:
            for i, (x1, y1) in enumerate(hidden_pos):
                for j, (x2, y2) in enumerate(output_pos):
                    w = w_ho[i % w_ho.shape[0], j]
                    color = 'red' if w < 0 else 'blue'
                    alpha = min(1.0, abs(w) / np.max(np.abs(w_ho)))
                    self.ax.plot([x1, x2], [y1, y2], color=color, alpha=alpha, linewidth=0.5)

        # Nodos de entrada
        for (x, y), a in zip(input_pos, input_activations):
            circle = patches.Circle((x, y), 0.015, color=plt.cm.Greys(a))
            self.ax.add_patch(circle)

        # Nodos ocultos
        for (x, y), a in zip(hidden_pos, hidden_activations):
            circle = patches.Circle((x, y), 0.02, color=plt.cm.viridis(a))
            self.ax.add_patch(circle)

        # Nodos de salida
        for idx, ((x, y), a) in enumerate(zip(output_pos, output_activations)):
            circle = patches.Circle((x, y), 0.03, color=plt.cm.plasma(a))
            self.ax.add_patch(circle)
            self.ax.text(x + 0.02, y, f"{idx}: {a:.2f}", va='center', fontsize=8)

        self.ax.set_title("Red Neuronal Simplificada (en tiempo real)", fontsize=14)
        self.canvas.draw()
