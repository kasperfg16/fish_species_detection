import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

from collections import OrderedDict

import model_functions
import processing_functions

import json

def load_predition_model(checkpoint):
    # Load in a mapping from category label to category name
    class_to_name_dict = processing_functions.load_json('classes_dictonary.json')

    if torch.cuda.is_available():
        map_location = torch.device('cuda')
        device = 'cuda'

    else:
        map_location = torch.device('cpu')
        device = 'cpu'
    
    # Load pretrained network
    model = model_functions.load_checkpoint(checkpoint, map_location)

    checkpoint = torch.load(checkpoint, map_location=map_location)

    return checkpoint, model, class_to_name_dict, device


def predict_species(image_dir, topk, checkpoint, model, class_to_name_dict, device):

    # Scales, crops, and normalizes a PIL image for the PyTorch model; returns a Numpy array
    image = processing_functions.process_image(image_dir, checkpoint['hidden_layer_units'])

    # Display image
    processing_functions.imshow(image)

    # Highest k probabilities and the indices of those probabilities corresponding to the classes (converted to the actual class labels)
    probabilities, classes = model_functions.predict(image_dir, model, checkpoint['hidden_layer_units'], device, topk=topk)  

    print(probabilities)
    print(classes)

    # Display the image along with the top 5 classes
    processing_functions.display_image(image_dir, class_to_name_dict, classes, checkpoint['hidden_layer_units'], probabilities)
    
    prediction = classes[0]

    return prediction