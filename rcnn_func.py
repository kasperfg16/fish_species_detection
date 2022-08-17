import os
import numpy as np
import torch
import torchvision
import cv2
import utils
import transforms as T
from PIL import Image
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from engine import train_one_epoch, evaluate
from matplotlib import pyplot as plt

"""
class FishDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        self.path_images = None
        self.path_masks = None
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PennFudanPed/PNGImages/"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PennFudanPed/PedMasks/"))))

    def __getitem__(self, idx):
        # load images and masks
        img_path = os.path.join(self.root, self.path_images, self.imgs[idx])
        mask_path = os.path.join(self.root, self.path_masks, self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = Image.open(mask_path)
        # convert the PIL Image into a numpy array
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

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        isCod = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["isCod"] = isCod

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)
"""

class PennFudanDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))

    def __getitem__(self, idx):
        # load images ad masks
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
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


def run_rcnn_trainer(arguments, masksPath, masksPathOther):
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # our dataset has two classes only - background and person
    num_classes = 2
    # use our dataset and defined transformations
    dataset = PennFudanDataset('PennFudanPed', get_transform(train=True))
    dataset_test = PennFudanDataset('PennFudanPed', get_transform(train=False))

    # split the dataset in train and test set
    indices = torch.randperm(len(dataset.imgs)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-50])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)

    # get the model using our helper function
    model = get_model_instance_segmentation(num_classes)

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    # let's train it for 10 epochs
    num_epochs = 3

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)

    print("That's it!")

    print("Saving model to disk...")
    basedir = os.path.dirname(os.path.abspath(__file__))
    model_name = 'model.pth'
    model_path = os.path.join(basedir, model_name)

    torch.save(model.state_dict(), model_path)
    print("Model saved.")

    test_rcnn(dataset_test, model, device)


def test_rcnn(dataset_test, model, device):
    # pick one image from the test set
    img, _ = dataset_test[0]
    # put the model in evaluation mode
    model.eval()
    with torch.no_grad():
        prediction = model([img.to(device)])
        print(prediction)
    
    im_normal = Image.fromarray(img.mul(255).permute(1, 2, 0).byte().numpy())
    im_normal.show()
    im_mask = Image.fromarray(prediction[0]['masks'][0, 0].mul(255).byte().cpu().numpy())
    im_mask.show()

    plt.show(im_normal)
    plt.axis("off")
    plt.show()

    plt.show(im_mask)
    plt.axis("off")
    plt.show()

def validate_masks():
    folder = "fish_pics/rcnn_masks/annotations/images/"
    #folder = "fish_pics/PennFudanPed/PedMasks/"
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            norm_image = cv2.normalize(img, None, alpha=0, beta=256, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            cv2.imshow("Image", norm_image)
            cv2.waitKey(0)


# function to normalize a masks and save in a list
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
    
# function to save name of masks annotations in a text file
def save_annotations(imgs, bounding_box, imgs_names, label, path):
    counter = 0
    with open(path + "annotations.txt", 'w') as f:
        for img in imgs:
            write = "Image filename : " + "\"" + imgs_names[counter] + ".jpg" + "\"" + "\n" + \
            "Image size (X x Y x C) : " + "{} x {} x {}".format(img.shape[0], img.shape[1], img.shape[2]) + "\n" + \
            "Database : \"fish_database\"" + "\n" + \
            "Objects with ground truth : " + str(1) + " { " + "No idea what to put here lol" +" }" + "\n" + \
            "Original label for object 1 \"{}\" : ".format(label) + label + "\n" + \
            "Bounding box for object 1 \"{}\" (Xmin, Ymin) - (Xmax, Ymax) : ".format(label) + "({},{}) - ({}, {})".format(bounding_box[counter][0], bounding_box[counter][1], bounding_box[counter][2], bounding_box[counter][3]) + "\n" + \
            "Pixel mask for object 1 \"{}\" : ".format(label) + "\"" + (path + imgs_names[counter]) + ".png\""

            with open(path + imgs_names[counter] + ".txt", 'w') as f:
                f.write(write + '\n')

            counter += 1