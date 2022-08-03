import torch
import model_functions
import processing_functions

def load_predition_model(checkpoint_path):
    
    """
    Loads the prediction model for prediction (only needed once at startup).

    :param arguments: The arguments for the predict model
    :return: prediction model
    """

    print("Loading model...")

    # Load in a mapping from category label to category name
    class_to_name_dict = processing_functions.load_json()

    if torch.cuda.is_available():
        map_location = torch.device('cuda')
        device = 'cuda'

    else:
        map_location = torch.device('cpu')
        device = 'cpu'
    
    # Load pretrained network
    model, checkpoint = model_functions.load_checkpoint(checkpoint_path, map_location)

    print("Model loaded")

    return checkpoint, model, class_to_name_dict, device


def predict_species(img_list, topk, checkpoint, model, class_to_name_dict, device):

    # Display image
    predictions = []

    for n in img_list:

        # Highest k probabilities and the indices of those probabilities corresponding to the classes (converted to the
        # actual class labels)
        probabilities, classes = model_functions.predict(n, model, checkpoint['hidden_layer_units'], device, topk=topk)

        print(probabilities)
        print(classes)

        # Display the image along with the top 5 classes
        # processing_functions.display_image(n, class_to_name_dict, classes, checkpoint['hidden_layer_units'], probabilities)

        prediction = classes[0]
        predictions.append(prediction)

    return predictions
