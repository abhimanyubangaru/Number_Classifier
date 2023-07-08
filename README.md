# Number_Classifier
Created a LeNet model using Pytorch to classify digits 0-9. This project is about recognizing handwritten digits using a convolutional neural network (CNN) based on the LeNet architecture.

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

```shell
pip install numpy torch torchvision pillow matplotlib tkinter
```
## Installing
Clone the repository to your local machine:

```shell
git clone https://github.com/<your_username>/handwritten-digit-recognition.git
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


