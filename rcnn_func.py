import os
import torch
import torchvision
import cv2
import utils
import glob
import camera_cal as ccal
import transforms as T
import functions_openCV as fcv
import numpy as np
from codecs import escape_encode
from pickle import TRUE
from PIL import Image
from PIL import ImageChops
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from engine import train_one_epoch, evaluate
from os.path import exists


class FishDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "images"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "masks"))))

    def __getitem__(self, idx):
        # load images ad masks
        img_path = os.path.join(self.root, "images", self.imgs[idx])
        mask_path = os.path.join(self.root, "masks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = Image.open(mask_path)

        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target


def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                    hidden_layer,
                                                    num_classes)

    return model


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def run_rcnn_trainer(basedir, model_path, num_epochs):

    print("Running the mask RCNN trainer...")

    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # our dataset has two classes only - background and fish
    num_classes = 2
    # use our dataset and defined transformations
    dataset = FishDataset(basedir + '/fish_pics/rcnn_dataset', get_transform(train=True))
    dataset_test = FishDataset(basedir + '/fish_pics/rcnn_dataset', get_transform(train=False))

    # split the dataset in train and test set
    indices = torch.randperm(len(dataset.imgs)).tolist() 
    dataset = torch.utils.data.Subset(dataset, indices[:-50])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=True, num_workers=1,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=1,
        collate_fn=utils.collate_fn)

    model = get_model_instance_segmentation(num_classes)

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.001,
                                momentum=0.9, weight_decay=0.0005) # lr=0.005
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, num_epochs, print_freq=10,)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        torch.cuda.empty_cache()
        evaluate(model, data_loader_test, device=device)

    print("That's it!")

    print("Saving model to disk...")
    torch.save(model.state_dict(), model_path)
    print("Model saved.")

    test_rcnn(dataset_test, model_path)
    return dataset_test


def test_rcnn(basedir, model_path, use_morphology=False):

    # Get the right device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # our dataset has two classes only - background and fish
    model = get_model_instance_segmentation(2)

    # Move model to the right device
    model.to(device)

    # Check if device is on GPU
    if device.type == 'cuda':
        print("Running on the GPU")
        model.load_state_dict(torch.load(model_path))
    else:
        print("Running on the CPU")
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    dataset_test = FishDataset(basedir + '/fish_pics/rcnn_dataset', get_transform(train=False))
    indices = torch.randperm(len(dataset_test.imgs)).tolist()
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

    # Load in images and names so we can check which image it is we are testing
    images, image_names = load_images_from_folder(basedir + '/fish_pics/rcnn_dataset')

    count = 0
    contours = []
    img_normal = []
    image_names_list = []
    precisions = []
    labels = []
    # pick one image from the test set
    for img in dataset_test:
        img, _ = dataset_test[count]

        # put the model in evaluation mode
        model.eval()

        with torch.no_grad():
            prediction = model([img.to(device)])
            print(prediction)

        # Convert to PIL image type
        im_normal = Image.fromarray(img.mul(255).permute(1, 2, 0).byte().numpy())
        im_mask = Image.fromarray(prediction[0]['masks'][0, 0].mul(255).byte().cpu().numpy())

        # Get the precision
        precision = prediction[0]['scores'][0].item()
        precisions.append(precision)

        # print the precision
        print("Precision: " + str(precision))

        # Get the label
        label = prediction[0]['labels'][0].item()
        if label == 1:
            label = "fish"
        labels.append(label)

        # Print the label
        print("Label: " + str(label))
        
        # Find the image name by brute force, not the best method, but it works
        img_names = find_image_name(im_normal, images, image_names)
        image_names_list.append(img_names)

        # Convert from PIL image type to cv2 image type
        open_cv_image_normal = np.array(im_normal) 
        open_cv_image_normal = open_cv_image_normal[:, :, ::-1].copy() 
        open_cv_image_normal = cv2.flip(open_cv_image_normal, 1)
        open_cv_image_mask = np.array(im_mask) 
        open_cv_image_mask = open_cv_image_mask[:, ::-1].copy() 

        if use_morphology:
            # Do some morphological operations to get rid of noise
            kernel = np.ones((5,5),np.uint8)
            open_cv_image_mask_morph = cv2.erode(open_cv_image_mask, kernel, iterations=1)

            contour, __ = cv2.findContours(open_cv_image_mask_morph, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        else:
            contour, __ = cv2.findContours(open_cv_image_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
        cv2.drawContours(open_cv_image_normal, contour, -1, (0,255,0), 1)
        cv2.imwrite(basedir + "/fish_pics/rcnn_dataset/validation/" + "final_image_contour_" + img_names, open_cv_image_normal)
        
        contours.append(contour)
        img_normal.append(open_cv_image_normal)

        count += 1

    return image_names_list, img_normal, contours, precisions, labels


def load_images_from_folder(folder):
    image_list = []
    image_names = []
    for filename in glob.glob(folder + "/images/*.png"): #assuming png
        im=Image.open(filename)
        filename = filename.replace(folder + '/images\\', '')
        image_names.append(filename)
        image_list.append(im)

    return image_list, image_names


def find_image_name(img, img_set, names):

    count = 0

    for im in img_set:
        diff = ImageChops.difference(img, im)

        if not diff.getbbox():
            print("Found name: " + names[count])
            return names[count]
        
        count += 1


def validate_masks(arguments, path):
    if not arguments.google_colab:
        folder = path
        for filename in os.listdir(folder):
            img = cv2.imread(os.path.join(folder,filename))
            if img is not None:
                norm_image = cv2.normalize(img, None, alpha=0, beta=256, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                cv2.imshow("Image", norm_image)
                cv2.waitKey(0)
    else:
        print("Using google colab does not allow for images to show. Saving images inside validation folder instead.")
        folder = path
        for filename in os.listdir(folder):
            img = cv2.imread(os.path.join(folder,filename))
            if img is not None:
                norm_image = cv2.normalize(img, None, alpha=0, beta=256, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                cv2.imwrite(arguments.validation_folder + filename + "_validation.jpg", norm_image)


def normalize_masks(masks):
    normalized_masks = []
    for mask in masks:
        # First convert to black and white image
        gray_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        # Then convert to black and white image and then normalize image to turn it into a bit image
        height, width = gray_mask.shape
        for i in range(height):
            for j in range(width):
                # img[i, j] is the RGB pixel at position (i, j)
                # check if it's [0, 0, 0] and replace with [255, 255, 255] if so
                if gray_mask[i, j].sum() != 0:
                    gray_mask[i, j] = 1

        #norm_image = cv2.normalize(gray_mask, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        normalized_masks.append(gray_mask)
    return normalized_masks
    

def save_annotations(imgs, bounding_box, imgs_names, label, path):
    counter = 0

    for img in imgs:
        write = "Image filename : " + "\"" + imgs_names[counter] + ".jpg" + "\"" + "\n" + \
        "Image size (X x Y x C) : " + "{} x {} x {}".format(img.shape[0], img.shape[1], img.shape[2]) + "\n" + \
        "Database : \"fish_database\"" + "\n" + \
        "Objects with ground truth : " + str(1) + " { " + "No idea what to put here lol" +" }" + "\n" + \
        "Original label for object 1 \"{}\" : ".format(label) + label + "\n" + \
        "Bounding box for object 1 \"{}\" (Xmin, Ymin) - (Xmax, Ymax) : ".format(label) + "({},{}) - ({}, {})".format(bounding_box[counter][0], bounding_box[counter][1], bounding_box[counter][2], bounding_box[counter][3]) + "\n" + \
        "Pixel mask for object 1 \"{}\" : ".format(label) + "\"" + (path + imgs_names[counter]) + ".png\""

        with open(path + '/' + imgs_names[counter] + ".txt", 'w') as f:
            f.write(write + '\n')

            counter += 1


def create_dataset_classification(arguments, imgs, fish_names, fish_masks, bounding_boxes, label, path_images, path_annotation):

    print("Creating dataset...")

    # Save normalized masks images in a folder
    normalized_masks = normalize_masks(fish_masks)
    counter = 0
    for mask in normalized_masks:
        cv2.imwrite(path_images + fish_names[counter] + ".png", mask)
        counter += 1
    
    save_annotations(imgs, bounding_boxes, fish_names, label, path_annotation)

    print("Done creating dataset!")


def save_dataset_mask_rcnn(imgs, fish_names, fish_masks, bounding_boxes, label, path_masks, path_annotations, path_imgs):

    # Save images in a folder
    counter = 0
    for img in imgs:
        path_img = os.path.join(path_imgs, fish_names[counter])
        path_img = path_img + '.png'
        print(path_img)
        cv2.imwrite(path_img, img)
        counter += 1

    # Save masks in a folder
    counter = 0
    normalized_masks = normalize_masks(fish_masks)
    for mask in normalized_masks:
        path_img = os.path.join(path_masks, fish_names[counter])
        path_img = path_img + '.png'
        print(path_img)
        cv2.imwrite(path_img, mask)
        counter += 1

    save_annotations(imgs, bounding_boxes, fish_names, label, path_annotations)


def create_dataset_mask_rcnn(arguments):

    load_folder = arguments.image_dir

    # Find path to folder where "train.py" python file is
    # Insures that we can run the script from anywhere and it will still work
    basedir = os.path.dirname(os.path.abspath(__file__))
    name_classes_folder = load_folder
    path_classes_folder = basedir + name_classes_folder

    # Find classes
    classes = next(os.walk(path_classes_folder))[1]

    for _class in classes:
        path_class_folder = os.path.join(path_classes_folder, _class)
        _, _, img_names = next(os.walk(path_class_folder))
        for img_name in img_names:
            if not arguments.undistorted:
                
                # Find a path to each images
                path_img = os.path.join(path_class_folder, img_name)
                img_list_fish = []
                imgs = []

                # Find image name with no extention e.g. ".png"
                img_name_no_ex = os.path.splitext(img_name)[0]
                img_list_fish.append(img_name_no_ex)
                img = cv2.imread(path_img)
                imgs.append(img)

                # Undistort images
                dst = ccal.undistort_imgs(imgs)

                # Isolate fish contours
                isolatedFish, contoursFish, masks, bounding_boxes = fcv.isolate_fish(dst, img_list_fish, display=False)

                isolatedFish = fcv.resize_imgs(isolatedFish, 10)
                imgs = fcv.resize_imgs(dst, 10)

                # Create paths for folders
                path_dataset = '/fish_pics/rcnn_dataset/images'

                imgs_folder = '/fish_pics/rcnn_dataset/images'
                path_imgs_folder = basedir + imgs_folder

                masks_folder = '/fish_pics/rcnn_dataset/masks'
                path_masks_folder = basedir + masks_folder

                annotations_folder = '/fish_pics/rcnn_dataset/annotations'
                path_annotations_folder = basedir + annotations_folder

                # Delete old dataset if it exists 
                # if exists(path_imgs_folder):
                #    shutil.rmtree(path_imgs_folder)
                
                # Create folders
                if not exists(path_imgs_folder):
                    os.makedirs(path_imgs_folder)
                
                if not exists(path_masks_folder):
                    os.makedirs(path_masks_folder)
                
                if not exists(path_annotations_folder):
                    os.makedirs(path_annotations_folder)

                # Save dataset
                save_dataset_mask_rcnn(
                    imgs=imgs,
                    fish_names=img_list_fish,
                    fish_masks=isolatedFish,
                    bounding_boxes=bounding_boxes,
                    label=_class,
                    path_masks=path_masks_folder,
                    path_annotations=path_annotations_folder,
                    path_imgs=path_imgs_folder)

        validate_masks(arguments, "fish_pics/rcnn_dataset/masks/")

    print("Done creating dataset!")