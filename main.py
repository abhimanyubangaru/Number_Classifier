# Built-in imports
import tkinter as tk
from tkinter import Canvas, Button

# Third-party imports
from PIL import Image, ImageDraw, ImageOps
import matplotlib.pyplot as plt
import numpy as np
import torch 
from torch import nn
from torchvision.transforms import ToTensor
from model import create_lenet, inference

# Constants
WIDTH, HEIGHT = 280, 280
WHITE, BLACK = (255, 255, 255), (0, 0, 0)
WEIGHTS_PATH = 'updated_model_weights.pth'

def load_model(weights_path):
    """Load the model from weights."""
    try:
        model = create_lenet()
        state_dict = torch.load(weights_path)
        model.load_state_dict(state_dict)
        return model
    except FileNotFoundError:
        print(f"No weights file found at {weights_path}")
        return None

def predict_digit(img, model):
    """Preprocess image and predict digit."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    img = img.resize((28,28))
    img_t = ToTensor()(img).unsqueeze(0)
    img_t = img_t.to(device)
    model.to(device)
    outputs = model(img_t)
    _, predicted = outputs.max(1)
    prob = nn.functional.softmax(outputs, dim=1)[0] * 100
    return prob.detach().cpu().numpy()

# Try to load the model
model = load_model(WEIGHTS_PATH)
if model is None:
    exit()

# Create GUI
window = tk.Tk()
window.title("Digit Classifier")
image = Image.new("RGB", (WIDTH, HEIGHT), WHITE)
draw = ImageDraw.Draw(image)
canvas = Canvas(window, width=WIDTH, height=HEIGHT, bg='white')
canvas.pack()
canvas.old_coords = None

def start_drawing(e):
    canvas.old_coords = e.x, e.y

def stop_drawing(e):
    canvas.old_coords = None

def draw_line(e):
    if canvas.old_coords:
        x, y = e.x, e.y
        x1, y1 = canvas.old_coords
        canvas.create_line(x, y, x1, y1, fill="black", width=5)
        draw.line([x, y, x1, y1], fill="black", width=5)
        canvas.old_coords = x, y

# Bind mouse button events to the corresponding functions on the canvas
canvas.bind("<Button-1>", start_drawing)
canvas.bind("<ButtonRelease-1>", stop_drawing)
canvas.bind("<B1-Motion>", draw_line)

def handle_predict():
    """Handle the predict button click."""
    img = image.resize((28,28))
    img = ImageOps.invert(img)
    img = img.convert(mode='L')
    result = predict_digit(img, model)
    plt.bar(np.arange(10), result)
    plt.xlabel('Digits')
    plt.ylabel('Probabilities')
    plt.show()

def clear():
    """Handle the clear button click."""
    canvas.delete('all')
    draw.rectangle([0, 0, WIDTH, HEIGHT], fill=WHITE)
    last_point = None

# Buttons
btn_clear = Button(text="Clear", command=clear)
btn_clear.pack(side='bottom')
btn_predict = Button(text="Predict", command=handle_predict)
btn_predict.pack(side='bottom')

window.mainloop()
