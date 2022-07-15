import json
import argparse
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

run = True
while run:

    print("start")
    # Capture photo
    img = cv2.imread("C:/Users/Kaspe/OneDrive/Onenote/GitHub/fish_species_detection/data_set/test/cod/GOPR0014.JPG")
    cv2.waitKey(0)
    cv2.imshow("img", img)
    cv2.waitKey(0)

    # Predict species
    

    # Segment sea creature
    mask_cod, segmented_images = ftc.segment_OPENCV(img)
    cv2.imshow("mask_cod", mask_cod)
    cv2.waitKey(0)

    # Measure length