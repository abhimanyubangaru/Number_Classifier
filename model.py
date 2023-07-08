import torch, torchvision 
from torch import nn
from torch import optim 
from torchvision.transforms import ToTensor
import torch.nn.functional as F
import matplotlib.pyplot as plt
import copy

from sklearn.metrics import confusion_matrix
import pandas as pd 
import numpy as np 

import requests
from PIL import Image 
from io import BytesIO

# num_batch = 64

# T = torchvision.transforms.Compose([
#     torchvision.transforms.ToTensor()
# ])
# train_data = torchvision.datasets.MNIST('mnist_data',train=True, download=True, transform = T)
# validation_data = torchvision.datasets.MNIST('mnist_data',train=False, download=True, transform = T)

# train_dl = torch.utils.data.DataLoader(train_data,batch_size = num_batch)
# val_dl = torch.utils.data.DataLoader(validation_data,batch_size = num_batch)

# plt.imshow(train_data[100][0][0])

def create_lenet():
    model = nn.Sequential(
        
        nn.Conv2d(1,6,5,padding=2),
        nn.ReLU(),
        nn.AvgPool2d(2, stride=2),
        
        nn.Conv2d(6,16,5,padding=0),
        nn.ReLU(),
        nn.AvgPool2d(2, stride=2),
        
        nn.Flatten(),
        nn.Linear(16*5*5,120),
        nn.ReLU(),
        nn.Linear(120,84), 
        nn.ReLU(),
        nn.Linear(84,10)
        
    )
    return model

def validate(model, data):
    correct = 0
    for i, (images, labels) in enumerate(data):
        images = images.to("cpu")  # Move images to CPU
        x = model(images)
        value, pred = torch.max(x, 1)
        correct += (pred == labels).sum().item()
    accuracy = 100 * correct / len(data.dataset)
    return accuracy

def train(num_epoch=3, lr=1e-3,device="cpu"):
    accuracies = []
    cnn = create_lenet().to(device)
    cec = nn.CrossEntropyLoss()
    optimizer = optim.Adam(cnn.parameters(),lr=lr)
    max_accuracy = 0
    
    for epoch in range(num_epoch):
        for i, (images,labels) in enumerate(train_dl):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            pred = cnn(images)
            loss = cec(pred, labels)
            loss.backward()
            optimizer.step()
        accuracy = float(validate(cnn, val_dl))
        accuracies.append(accuracy)
        if accuracy > max_accuracy:
            best_model = copy.deepcopy(cnn)
            max_accuracy = accuracy
            print("Saving with Model with Best accuracy", accuracy)
        print("Epoch", epoch+1, "Accuracy", accuracy,"%")
    plt.plot(accuracies)
    return best_model
    
def predict_dl(model,data):
    y_pred = []
    y_true = [] 
    for i, (images,labels) in enumerate(data):
        x = model(images)
        value, pred = torch.max(x,1)
        y_pred.extend(list(pred.numpy()))
        y_true.extend(list(labels.numpy()))
    return np.array(y_pred),np.array(y_true)

# def inference(path, model, device):
#     r = requests.get(path)
#     with BytesIO(r.content) as f:
#         image = Image.open(f).convert(mode="L")
#         image = image.resize((28,28)) #model only accepts 28 by 28 
#         x = (255 - np.expand_dims(np.array(image), -1))/255.
#     with torch.no_grad():
#         pred = model(torch.unsqueeze(T(x), axis=0).float().to(device))
#         return F.softmax(pred, dim=-1).cpu().numpy()

def inference(img, model, device):
    img = img.resize((28,28))
    x = (255 - np.array(img))/255.
    x = torch.unsqueeze(ToTensor()(x), axis=0).float().to(device)
    with torch.no_grad():
        pred = model(x)
        return F.softmax(pred, dim=-1).cpu().numpy()
    
def plot_predictions(pred):
    # Create an array with the number of classes
    classes = np.arange(len(pred))

    # Create a bar plot
    plt.bar(classes, pred)

    # Add title and labels
    plt.title('Prediction Probabilities')
    plt.xlabel('Classes')
    plt.ylabel('Probability')

    # Show the plot
    plt.show()
