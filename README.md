# Fish_species_detection

Repo for fish species detection project at Aalborg University

## Perquisites

- This is developed for windows and only tested on Window 11 OS

## Setup

1. Download the repository:

    ``` bash
    https://github.com/kasperfg16/fish_species_detection.git
    ```

2. Install conda

    <https://developers.google.com/earth-engine/guides/python_install-conda#windows>

3. Open command prompt

    - a)

        Install conda:

        <https://developers.google.com/earth-engine/guides/python_install-conda#windows>

    - b)

        Navigate to the repository folder (...\fish_species_detection)

        Example:

        ``` bash
        cd C:\Users\Kaspe\OneDrive\Onenote\GitHub\fish_species_detection
        ```

    - c)

        Create a conda environment

        ``` bash
        conda create -n env_pytorch python=3.10.4
        ```

    - d)

        Activate the environment:

        ``` bash
        conda activate env_pytorch
        ```

        Install required packages:

        ``` bash
        pip install -r requirements.txt
        ```

    Now you're ready to run the code

## Use

To train the model put images into fish_pics\input_images

## Method used

1. A dataset of 310 sea creatures including 34 different species where made using a custom made photobox.
    - For each sea creature the dataset contains:
        - The lenght of the sea creature
        - The species of the sea creature
        - Two images from different sides of the sea creature

2. Pytorch is used along with a pre-trained model (VGG16 - <https://pytorch.org/vision/main/models/generated/torchvision.models.vgg16.html>). The model pre-trained model containing weights that has been obtained by training it using the IMAGENET1K_V1 dataset. The output of the pre-trained model is given to a classifier that is trained on the data set created in this project. The final model is thereby a pre-trained model combined with a classifier. The input to the model is the .png image taken directly with a GoPro camera. The output of the final model is the k-most probable species.

As of 03/07/2022, The classifier have only been trained on two classes i.e. "cod" and "other"

3.

1. Use a checker board for calibration <https://github.com/opencv/opencv/blob/4.x/doc/pattern.png>

## Train your own classifier on your own dataset with google colab

Follow the link and follow the instructions

<https://colab.research.google.com/drive/1qksJvIxpnAULe_8XfI04jG13a-3klOKA?usp=sharing>

## Inspiration used to create this project

1. loading of images:

    <https://ryanwingate.com/intro-to-machine-learning/deep-learning-with-pytorch/loading-image-data-into-pytorch/>

2. <https://docs.microsoft.com/en-us/learn/modules/intro-computer-vision-pytorch/2-image-data>

3. Using a pretrained model to perform image classification

    <https://github.com/LeanManager/PyTorch_Image_Classifier/blob/master/train.py>

    Youtube video: <https://www.youtube.com/watch?v=zFA8Cm13Xmk&t=513s>
