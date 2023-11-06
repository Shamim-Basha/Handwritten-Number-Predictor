# Handwritten Number Predictor

The Handwritten Number Predictor is a simple application that uses Python, Pygame, and TensorFlow to predict handwritten numbers. This project is designed for learning and demonstrating how to build a basic image classification model and integrate it into a Pygame application. You can use and modify this code as you see fit.

## Prerequisites

Before running the Handwritten Number Predictor, make sure you have the following dependencies installed on your system:

### Python

You can download and install Python from the official website: [Python Downloads](https://www.python.org/downloads/)

### Pygame

You can install Pygame using pip:

```bash
pip install pygame
```

### TensorFlow

You can install TensorFlow using pip:

```bash
pip install tensorflow
```

## How to Use

1. Draw a single digit (0-9) in the drawing area.

2. Press the **Predict** button to see the model's prediction for the drawn digit.

3. You can also clear the drawing by pressing the **Reset** button.

## Model Details

The model used for predicting handwritten digits in this application is a Convolutional Neural Network (CNN) trained on the MNIST dataset. The model is loaded from the `model.h5` file, which should be located in the same directory as the application. You can replace this model with your own trained model if desired.

## Customization

You can customize various aspects of the application by editing the `config.py` file. Here are some things you can change:

- Screen dimensions (width and height)
- Drawing area dimensions
- Colors and fonts

Feel free to experiment and tweak the application to your liking!

## Contributing

If you'd like to contribute to this project, please feel free to fork the repository and create a pull request. We welcome any improvements, bug fixes, or new features.

## License

This Handwritten Number Predictor is released under the [MIT License](LICENSE). You are free to use, modify, and distribute it for your own purposes.

## Acknowledgments

This project was created for educational purposes, and we would like to thank the Pygame and TensorFlow communities for their support and the open-source community for their contributions to the Python ecosystem.

If you have any questions or need assistance, please don't hesitate to contact us.

Enjoy predicting handwritten numbers! 🖋🔢
