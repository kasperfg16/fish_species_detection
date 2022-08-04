import json
import os, os.path
import torch
import processing_functions
from torch import nn
from torchvision.models import VGG16_Weights, AlexNet_Weights
from torchvision import models
from collections import OrderedDict
from workspace_utils import active_session

# Function for saving the model checkpoint
def save_checkpoint(model, training_dataset, arch, epochs, lr, hidden_units, input_size):

    model.class_to_idx = training_dataset.class_to_idx

    checkpoint = {'input_size': (3, 224, 224),
                  'output_size': 102,
                  'hidden_layer_units': hidden_units,
                  'batch_size': 64,
                  'learning_rate': lr,
                  'model_name': arch,
                  'model_state_dict': model.state_dict(),
                  'epochs': epochs,
                  'class_to_idx': model.class_to_idx,
                  'clf_input': input_size}

    torch.save(checkpoint, 'checkpoint.pth')
    
# Function for loading the model checkpoint    
def load_checkpoint(checkpoint_path, map_location):

    if not os.path.isfile(checkpoint_path):
        print('Your path to the .pth file did not exist. Trying to find the file checkpoint.pth file instead')
        basedir = os.path.dirname(os.path.abspath(__file__))
        path_img_folder = '/checkpoint.pth'
        checkpoint_path = basedir + path_img_folder

    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    
    if checkpoint['model_name'] == 'vgg':
        model = models.vgg16(weights=VGG16_Weights.DEFAULT)
        
    elif checkpoint['model_name'] == 'alexnet':  
        model = models.alexnet(weights=AlexNet_Weights.DEFAULT)
    else:
        print("Architecture not recognized.")
        
    for param in model.parameters():
            param.requires_grad = False    
    
    model.class_to_idx = checkpoint['class_to_idx']

    basedir = os.path.dirname(os.path.abspath(__file__))
    path_img_folder = '/fish_pics/input_images/'
    image_dir = basedir + path_img_folder

    subfolders = next(os.walk(image_dir))[1]

    num_classes = len(subfolders)

    classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(checkpoint['clf_input'], checkpoint['hidden_layer_units'])),
                                        ('relu', nn.ReLU()),
                                        ('drop', nn.Dropout(p=0.5)),
                                        ('fc2', nn.Linear(checkpoint['hidden_layer_units'], num_classes)),
                                        ('output', nn.LogSoftmax(dim=1))]))

    model.classifier = classifier
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, checkpoint


# Function for the validation pass
def validation(model, validateloader, criterion, device):
    
    val_loss = 0
    accuracy = 0
    
    for images, labels in iter(validateloader):

        images, labels = images.to(device), labels.to(device)

        output = model.forward(images)
        val_loss += criterion(output, labels).item()

        probabilities = torch.exp(output)
        
        equality = (labels.data == probabilities.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    
    return val_loss, accuracy


# Function for measuring network accuracy on test data
def test_accuracy(model, test_loader, device):

    # Do validation on the test set
    model.eval()

    if device == 'cuda':
        if not torch.cuda.is_available():
            print("Could not find cuda enabled GPU")
            device = 'cpu'
    else: 
        device = 'cpu'

    print("Device used for classification: ", device)
    model.to(device)

    with torch.no_grad():
    
        accuracy = 0
    
        for images, labels in iter(test_loader):
    
            images, labels = images.to(device), labels.to(device)
    
            output = model.forward(images)

            probabilities = torch.exp(output)
        
            equality = (labels.data == probabilities.max(dim=1)[1])
        
            accuracy += equality.type(torch.FloatTensor).mean()
        
        print("Test Accuracy: {}".format(accuracy/len(test_loader)))    

        acc = accuracy/len(test_loader)
    return acc
        

# Train the classifier
def train_classifier(model, optimizer, criterion, arg_epochs, train_loader, validate_loader, device):

    epochs = arg_epochs
    steps = 0
    print_every = 3

    if device == 'cuda':
        if not torch.cuda.is_available():
            print("Could not find cuda enabled GPU")
            device = 'cpu'
        else:
            print("Device used for training: ", torch.cuda.get_device_name())
    else: 
        device = 'cpu'

    print("Type of device used for training: ", device)
    model.to(device)

    # Check if user specified number of epochs or set to -1 for validation accuracy to reach is 100%
    if not arg_epochs == -1:
        num_epochs = arg_epochs
    else:
        num_epochs = 1

    Val_acc = 0
    epoch = 0

    # Run while 'ctrl+c' is not pressed
    try:
        # Either run until validation accuracy is 100% or until specified epoch number is reached.
        while not Val_acc == 1:
            torch.cuda.empty_cache()
            for e in range(num_epochs):
                epoch += 1
                model.train()

                running_loss = 0

                for images, labels in iter(train_loader):
            
                    steps += 1
            
                    images, labels = images.to(device), labels.to(device)
            
                    optimizer.zero_grad()
            
                    output = model.forward(images)
                    loss = criterion(output, labels)
                    loss.backward()
                    optimizer.step()
            
                    running_loss += loss.item()
            
                    if steps % print_every == 0:
                    
                        model.eval()
                    
                        # Turn off gradients for validation, saves memory and computations
                        with torch.no_grad():
                            validation_loss, accuracy = validation(model, validate_loader, criterion, device)
                
                        Val_acc = accuracy/len(validate_loader)
                        if arg_epochs == -1:
                            print("Epoch: {} ".format(epoch))
                        else:
                            print("Epoch: {}/{}.. ".format(e+1, epochs))

                        print(
                            "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                            "Validation Loss: {:.3f}.. ".format(validation_loss/len(validate_loader)),
                            "Validation Accuracy: {:.3f}".format(Val_acc))

                        running_loss = 0 

    except KeyboardInterrupt:
                pass
                    
def predict(image, model, hidden_size, device, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''

    image = processing_functions.process_image(image, hidden_size)

    # Convert image to PyTorch tensor
    if device == 'cuda':
        if not torch.cuda.is_available():
            print("Could not find cuda enabled GPU using cpu instead")
            device = 'cpu'
    
    image = torch.from_numpy(image).type(torch.FloatTensor)

    print("Device used for classification: ", device)
    model.to(device)

    # Returns a new tensor with a dimension of size one inserted at the specified position.
    image = image.unsqueeze(0)
    if device == 'cuda':
        image = image.cuda()
    
    output = model.forward(image)

    probabilities = torch.exp(output)
    
    # Find the path to the .json file
    basedir = os.path.dirname(os.path.abspath(__file__))
    json_file_name = '/classes_dictonary.json'
    json_path = basedir + json_file_name

    # Opening JSON file
    f = open(json_path)

    # returns JSON object as
    # a dictionary
    data = json.load(f)

    # Get the number of classes
    classes = data.keys()
    num_classes = len(classes)
    num_classes = int(num_classes)
    f.close()

    if num_classes > topk:
        num_classes = topk

    # Probabilities and the indices of those probabilities corresponding to the classes
    top_probabilities, top_indices = probabilities.topk(num_classes)

    # Convert to lists
    top_probabilities = top_probabilities.detach().type(torch.FloatTensor).numpy().tolist()[0]
    top_indices = top_indices.detach().type(torch.FloatTensor).numpy().tolist()[0]

    # Convert topk_indices to the actual class labels using class_to_idx
    # Invert the dictionary so you get a mapping from index to class.

    idx_to_class = {value: key for key, value in model.class_to_idx.items()}

    top_classes = [idx_to_class[index] for index in top_indices]
    
    return top_probabilities, top_classes