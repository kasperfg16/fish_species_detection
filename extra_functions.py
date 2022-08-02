import os, os.path
import shutil
import json
import random

def make_data_sets(percent_train=80):
    '''
    # Divide images into train, test, and validation folders:
    '''

    # Find path to folder where "train.py" python file is
    basedir = os.path.dirname(os.path.abspath(__file__))
    path_img_folder = '/fish_pics/input_images/'
    img_folder_path = basedir + path_img_folder

    # Location of dataset
    data_dir = '/data_set'
    data_dir_path = basedir + data_dir

    # Remove old dataset
    if os.path.exists(data_dir_path):
        shutil.rmtree(data_dir_path)

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
                elif count <= num_train + num_test:
                    folder_name = "test"
                
                new_path = os.path.join(data_dir_path, folder_name, subfolder)
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
            json_file_name = "classes_dictonary.json"
            json_file_path = basedir + json_file_name
            with open(json_file_path, "w") as outfile:
                outfile.write(json_object)
            check = True
        else:
            with open(json_file_path) as f:
                data = json.load(f)
            
            data.update(addition)

            # Adding class names to classes_dictonary.json
            with open(json_file_path, 'w') as f:
                json.dump(data, f)

    # Create a validation folder
    folder_name = "validation"

    source_dir = img_folder_path
    valid_dir = os.path.join(data_dir_path, folder_name)
    shutil.copytree(source_dir, valid_dir)

    # Directories for "train" and "test"
    folder_name = "train"
    train_dir = os.path.join(data_dir_path, folder_name)
    folder_name = "test"
    test_dir = os.path.join(data_dir_path, folder_name)

    return num_classes, train_dir, valid_dir, test_dir