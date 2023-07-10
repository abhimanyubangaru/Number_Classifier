# Number_Classifier
Created a LeNet model using Pytorch to classify digits 0-9. This project is about recognizing handwritten digits using a convolutional neural network (CNN) based on the LeNet architecture. This was my first time using PyTorch, so I decided to use one of the simpler models.

## Why LeNet-5?

The LeNet-5 architecture is a classic and widely used architecture for image recognition, especially digit recognition tasks. The LeNet-5 architecture consists of two sets of convolutional layers and pooling layers, followed by a flattening convolutional layer, and finally two fully connected layers. This architecture is straightforward, easy to understand and serves as a good starting point for convolutional neural networks. 

## Model Performance

The model was trained on the MNIST dataset and achieved 99% accuracy on the validation data. This is a significant achievement as it ensures a high degree of reliability when predicting the digits in various handwritten instances.

## Training 
The model is trained using the train function in the model.py script. The function uses the Adam optimizer and the Cross-Entropy Loss. The training process also includes validation, and the model achieving the highest accuracy on the validation data is saved. The training was performed in a Jupyter notebook, which is included in the repository.

## Getting Started


### Prerequisites

You will need Python 3.6 or above, along with the following Python libraries installed:

- numpy
- torch
- torchvision
- PIL
- matplotlib
- tkinter

You can install these packages by running the following command:

## Installation

1. Clone the repository:
   `git clone https://github.com/your-username/digit-classifier.git`
2. Navigate to the cloned project directory.
3. Install the required packages by running the following command:
   `pip install -r requirements.txt`

## Setting up the conda environment 
Using the conda config file:
```
conda env create -f torch-conda-nightly.yml -n torch
```
To use this conda environment:
```
conda activate torch
```
## Running the Application

Navigate to the project directory and run:
```shell
python main.py
```


A GUI application will open up. Draw a digit in the provided canvas and click "Predict" to predict the digit. You can clear the canvas by clicking "Clear".

## File Structure

- `main.py`: This is the main script that you run to start the application. It sets up the GUI and handles the events.
- `model.py`: This contains the functions to create the CNN, validate its accuracy, and infer the digit from an image.
- `updated_model_weights.pth`: These are the pre-trained weights of the CNN, which are loaded into the model when you start the application.

## Contributing

I welcome contributions. Please reach me at abhi.bangaru24@gmail.com regarding any modifications to this program. Thank you!


