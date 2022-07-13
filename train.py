import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import torch
import torch.nn.functional as F
import model_functions
import processing_functions
import argparse
import extra_functions as ef

from torchvision import datasets, transforms, models
from collections import OrderedDict
from torch import optim
from torch import nn
from unittest import case

parser = argparse.ArgumentParser(description='Train Image Classifier')

# Command line arguments
parser.add_argument('--arch', type = str, default = 'vgg', help = 'NN Model Architecture')
parser.add_argument('--learning_rate', type = float, default = 0.001, help = 'Learning Rate')
parser.add_argument('--hidden_units', type = int, default = 10000, help = 'Neurons in the Hidden Layer')
parser.add_argument('--epochs', type = int, default = 20, help = 'Epochs')
parser.add_argument('--gpu', type = str, default = 'cuda', help = 'GPU or CPU')
parser.add_argument('--save_dir', type = str, default = 'checkpoint.pth', help = 'Path to checkpoint')

arguments = parser.parse_args()

# Image data directories
img_folder_path = 'images'

data_dir, num_classes = ef.make_data_sets(img_folder_path)

# Divide images into train, test, and validation folders:
train_dir = data_dir + '/train'
valid_dir = data_dir + '/validation'
test_dir = data_dir + '/test'

# Transforms for the training, validation, and testing sets
training_transforms, validation_transforms, testing_transforms = processing_functions.data_transforms()

# Load the datasets with ImageFolder
training_dataset, validation_dataset, testing_dataset = processing_functions.load_datasets(train_dir, training_transforms, valid_dir, validation_transforms, test_dir, testing_transforms)

# Using the image datasets and the trainforms, define the dataloaders
train_loader = torch.utils.data.DataLoader(training_dataset, batch_size=64, shuffle=True)
validate_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=32)
test_loader = torch.utils.data.DataLoader(testing_dataset, batch_size=32)

# Build and train the neural network (Transfer Learning)
if arguments.arch == 'vgg':
    input_size = 25088
    model = models.vgg16(pretrained=True)
elif arguments.arch == 'alexnet':
    input_size = 9216
    model = models.alexnet(pretrained=True)
    
print(model)

# Freeze pretrained model parameters to avoid backpropogating through them
for parameter in model.parameters():
    parameter.requires_grad = False

# Build custom classifier
classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(input_size, arguments.hidden_units)),
                                        ('relu', nn.ReLU()),
                                        ('drop', nn.Dropout(p=0.5)),
                                        ('fc2', nn.Linear(arguments.hidden_units, num_classes)),
                                        ('output', nn.LogSoftmax(dim=1))]))

model.classifier = classifier

# Loss function (since the output is LogSoftmax, we use NLLLoss)
criterion = nn.NLLLoss()

# Gradient descent optimizer
optimizer = optim.Adam(model.classifier.parameters(), lr=arguments.learning_rate)
    
model_functions.train_classifier(model, optimizer, criterion, arguments.epochs, train_loader, validate_loader, arguments.gpu)

model_functions.test_accuracy(model, test_loader, arguments.gpu)

model_functions.save_checkpoint(model, training_dataset, arguments.arch, arguments.epochs, arguments.learning_rate, arguments.hidden_units, input_size)