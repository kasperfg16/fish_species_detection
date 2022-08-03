from numpy import mean
import torch
import model_functions
import processing_functions
import argparse
import extra_functions as ef
from torchvision.models import AlexNet_Weights, VGG16_Weights
from torchvision import models
from collections import OrderedDict
from torch import optim
from torch import nn
from unittest import case

parser = argparse.ArgumentParser(description='Train Image Classifier')

# Command line arguments
parser.add_argument('--arch', type = str, default = 'vgg', help = 'NN Model Architecture')
parser.add_argument('--learning_rate', type = float, default = 0.001, help = 'Learning Rate')
parser.add_argument('--hidden_units', type = int, default = 10000, help = 'Neurons in the Hidden Layer')
parser.add_argument('--epochs', type = int, default = -1, help = 'Epochs. If epochs = -1 the training will run until the validation accuracy is = 100%')
parser.add_argument('--gpu', type = str, default = 'cuda', help = 'GPU or CPU')
parser.add_argument('--save_dir', type = str, default = 'checkpoint.pth', help = 'Path to checkpoint')
parser.add_argument('--percent_train', type = int, default = 80, help = 'How much of the data set should be used for training ((How much is used for testing) = 100% - percent_train)')
parser.add_argument('--num_of_k', type = int, default = 1, help = 'How many k\'s in k-fold validation')

arguments = parser.parse_args()

acc_list = []

for k in range(arguments.num_of_k):

    num_classes, train_dir, valid_dir, test_dir = ef.make_data_sets(arguments.percent_train)

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
        model = models.vgg16(weights = VGG16_Weights.IMAGENET1K_V1)
    elif arguments.arch == 'alexnet':
        input_size = 9216
        model = models.alexnet(weights = AlexNet_Weights.IMAGENET1K_V1)
        
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

    acc = model_functions.test_accuracy(model, test_loader, arguments.gpu)

    model_functions.save_checkpoint(model, training_dataset, arguments.arch, arguments.epochs, arguments.learning_rate, arguments.hidden_units, input_size)

    acc_list.append(acc)
    print('acc_list', acc_list)

print('acc_list', acc_list)

mean_acc = mean(acc_list)
print('mean_acc', mean_acc)

file1 = open("acc_k_fold_val.txt", "w") 
file1.write("acc_list = " + str(acc_list) + '\n' + 'mean_acc = ' + str(mean_acc))
file1.close()