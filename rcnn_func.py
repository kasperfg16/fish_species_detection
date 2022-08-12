import os
import numpy as np
import torch
import torchvision
from PIL import Image
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


class FishDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        self.path_images = None
        self.path_masks = None
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, self.path_images))))
        self.masks = list(sorted(os.listdir(os.path.join(root, self.path_masks))))

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

    def __len__(self):
        return len(self.imgs)


def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")

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


# function to normalize a masks and save in a list
def normalize_masks(masks):
    normalized_masks = []
    for mask in masks:
        normalized_masks.append(mask/255)
    return normalized_masks
    
# function to save name of masks in a text file
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