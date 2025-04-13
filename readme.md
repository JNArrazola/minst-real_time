# MNIST Real-Time Digit Recognizer with Neural Network Visualization

This is a PyQt6-based application that allows users to draw a digit on a 280x280 canvas. The system recognizes the digit in real time using a pre-trained MNIST model and visualizes the neural network structure and activations live.

## Features

- Real-time digit classification using a dense neural network trained on MNIST.
- Interactive canvas with smooth brush drawing.
- Live neural network visualization:
  - Input layer (compressed)
  - Hidden layer
  - Output layer with predicted digit highlighted
- Dynamic update of neuron activations while drawing.
- Optimized for performance using node and connection simplification.

## Requirements

- Python 3.9+
- PyQt6
- TensorFlow
- NumPy
- SciPy
- Matplotlib

Install dependencies with:

```bash
pip install -r requirements.txt
```

## File Structure

- `main.py`: Main GUI application that includes the drawing canvas and digit prediction logic.
- `visualizer.py`: Module responsible for rendering the neural network and updating its visualization in real time.
- `model.h5`: Pre-trained MNIST model in Keras HDF5 format (fully connected).
- `requirements.txt`: List of Python dependencies required to run the project.
- `model_metrics.py`: An independent module used to measure some model metrics like accuracy, averages, and the generation of the confusion matrix.

## Notes

- **All code comments are written in Spanish** for educational purposes.
- The neural network visualization simplifies the actual model structure for performance reasons:
  - 10 representative input nodes (from the 784 input features)
  - 16 hidden neurons
  - 10 output neurons corresponding to the digits 0–9
- Real-time prediction updates are throttled using a timer to avoid CPU overload while drawing.

## How to Run

Ensure that `model.h5` is present in the same directory. You can either provide your own pre-trained model or train one with the Keras MNIST dataset.

To start the application:

```bash
python main.py
```
