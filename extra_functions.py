import os, os.path, shutil
import shutil
import json
import random

def make_data_sets(img_folder_path, percent_train=80):
    '''
    # Divide images into train, test, and validation folders:
    '''

    # Location with images
    img_folder_path = "images/"

    # Location of dataset
    data_dir = "data_set"

    # Remove old dataset
    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)

    # Find subfolders in /images
    subfolders = next(os.walk(img_folder_path))[1]

    num_classes = len(subfolders)
    print("Number of classes: ", num_classes)

    check = False

    # For each subfolder, devide into train and test with given percentage.
    # Also create a JSON file with class labels
    for subfolder in subfolders:
        imgs_subfolder_path = img_folder_path + subfolder
        abs_path = os.path.abspath(imgs_subfolder_path)
        images = os.listdir(abs_path)

        # Shuffle list of images
        random.shuffle(images)

        num_images = len(images)

        num_train = round(num_images/100 * percent_train)
        num_test = round(num_images/100 * (100-percent_train))

        count = 1

        for image in images:
                if count <= num_train:
                    folder_name = "train"
                    new_path = os.path.join(data_dir, folder_name, subfolder)
                    if not os.path.exists(new_path):
                            os.makedirs(new_path)

                    old_image_path = os.path.join(imgs_subfolder_path, image)
                    new_image_path = os.path.join(new_path, image)
                    shutil.copy(old_image_path, new_image_path)

                elif count <= num_train + num_test:
                    folder_name = "test"
                    new_path = os.path.join(data_dir, folder_name, subfolder)
                    if not os.path.exists(new_path):
                            os.makedirs(new_path)

                    old_image_path = os.path.join(imgs_subfolder_path, image)
                    new_image_path = os.path.join(new_path, image)
                    shutil.copy(old_image_path, new_image_path)

                count += 1
        
        # Data to be written into JSON file
        addition ={subfolder : subfolder}

        if not check:
            # Serializing json 
            json_object = json.dumps(addition)
            # Writing to classes_dictonary.json
            with open("classes_dictonary.json", "w") as outfile:
                outfile.write(json_object)
            check = True
        else:
            with open('classes_dictonary.json') as f:
                data = json.load(f)
            
            data.update(addition)

            # Adding class names to classes_dictonary.json
            with open('classes_dictonary.json', 'w') as f:
                json.dump(data, f)

    # Create a validation folder
    folder_name = "validation"
    source_dir = img_folder_path
    destination_dir = os.path.join(data_dir, folder_name)
    shutil.copytree(source_dir, destination_dir)

    return data_dir, num_classes