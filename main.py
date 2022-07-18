from inspect import Parameter
import json
import argparse
from mimetypes import init
from tokenize import Double
import torch
import model_functions
import processing_functions
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb
import cv2
import functions_openCV as ftc
from torchvision import datasets, transforms, models
from torch import nn
from torch import optim
from collections import OrderedDict
import argparse
import predict

parser = argparse.ArgumentParser(description='Image Classifier Predictions')

# Command line arguments
parser.add_argument('--image_dir', type = str, default = "./images/other/GOPR0010.JPG", help = 'Path to image')
parser.add_argument('--checkpoint', type = str, default = 'C:/Users/Kaspe/OneDrive/Onenote/GitHub/fish_species_detection/checkpoint.pth', help = 'Path to checkpoint')
parser.add_argument('--topk', type = int, default = 5, help = 'Top k classes and probabilities')
parser.add_argument('--json', type = str, default = 'classes_dictonary.json', help = 'class_to_name json file')
parser.add_argument('--gpu', type = str, default = 'cuda', help = 'GPU or CPU')
parser.add_argument('--arUco_marker_cur', type = float, default = 19.2, help = 'ArUco marker circumference')

arguments = parser.parse_args()

init_cali = True
init_load_model = True

run = True
while run:

    # Photo with arUco marker
    aruco_marker_img = cv2.imread("C:/Users/Kaspe/OneDrive/Onenote/GitHub/fish_species_detection/arUco_in_box.JPG")

    # Capture photo
    img = cv2.imread(arguments.image_dir)
    
    # Make image fit screen and show image
    h_pixels, w_pixels, _= img.shape
    scale_to_screen = 5
    fit_to_screen = (int(w_pixels/scale_to_screen), int(h_pixels/scale_to_screen))
    ims = cv2.resize(img, (fit_to_screen))
    cv2.imshow("img", ims)
    cv2.waitKey(0)

    # Predict species

    ## Load model (only needed once at startup)
    if init_load_model:
        checkpoint, model, class_to_name_dict, device = predict.load_predition_model(arguments.checkpoint)
        print("Model loaded")
        init_load_model = False
    
    ## Predict
    prediciton = predict.predict_species(arguments.image_dir, arguments.topk, checkpoint, model, class_to_name_dict, device)

    # Measure length of sea creature

    ## Segment sea creature
    mask_cod, segmented_images = ftc.segment_OPENCV(img)

    ## Undistort image
    
    
    ## Use AruCo markers for size estimation

    ### Load ArUco detector and calibrate object size estimation (only needed once at startup)
    if init_cali:
        parameters = cv2.aruco.DetectorParameters_create()
        aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_50)

        corners, _, _ = cv2.aruco.detectMarkers(aruco_marker_img, aruco_dict, parameters=parameters)

        if not corners:
            print("No arUco markers found, try again")
            continue
        elif corners:
            int_corners = np.int0(corners)
            cv2.polylines(aruco_marker_img, int_corners, True, (0,255,0), 5)

            ### ArUco parimiter
            aruco_perimiter = cv2.arcLength(corners[0], True)

            ### Pixel to cm ratio
            pixel_cm_ratio = aruco_perimiter / arguments.arUco_marker_cur

            print("Camera setup calibrated")
            init_cali = False
    
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 80, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    biggest_cnt = max(contours, key = cv2.contourArea)

    rect = cv2.minAreaRect(biggest_cnt)
    (x,y), (w,h), angle = rect

    ### Get width and height of objects in cm
    w_cm = round(w / pixel_cm_ratio,2)
    h_cm = round(h / pixel_cm_ratio,2)

    box = cv2.boxPoints(rect)
    box = np.int0(box)

    cv2.circle(img, (int(x),int(y)), 5 ,(0, 0, 255), -1)

    cv2.polylines(img, [box], True, (255, 0, 0), 2)

    cv2.putText(img, "Width {} cm".format(w_cm, 1), (int(x + 10),int(y - 50)), cv2.FONT_HERSHEY_PLAIN, 5, (100, 200, 0), 5)
    cv2.putText(img, "Height {} cm".format(h_cm, 1), (int(x + 10),int(y + 50)), cv2.FONT_HERSHEY_PLAIN, 5, (100, 200, 0), 5)
    cv2.putText(img, "Species: {}".format(prediciton, 1), (int(x + 10),int(y + 150)), cv2.FONT_HERSHEY_PLAIN, 5, (100, 200, 0), 5)

    ims = cv2.resize(img, (fit_to_screen))
    cv2.imshow("img", ims)
    cv2.waitKey(0)