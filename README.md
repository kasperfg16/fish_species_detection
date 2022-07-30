# Fish_species_detection
Repo for fish species detection project at Aalborg University

## Perquisites

- This is developed for windows and only tested on Window 10 OS

## Setup

1. Download the repository:

    ``` bash
    https://github.com/kasperfg16/fish_species_detection.git
    ```

2. Open powershell as administrator

    - a) 
    
        Run following command:

        ``` bash
        New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" `
        -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
        ```

    - b)
    
        Close the powershell window and reboot your computer


2. Open command prompt

    - a) 
    
        Navigate to the repository folder (...\fish_species_detection)

        Example:

        ``` bash
        cd C:\Users\Kaspe\OneDrive\Onenote\GitHub\fish_species_detection
        ```

    - b)
    
        Create a conda environment

        Create a conda environment and install required packages:

        ``` bash
        conda create -n env_pytorch python=3.10.4
        ```
    
    - c)
        
        Activate the environment using:

        ``` bash
        conda activate env_pytorch
        ```

        Install required packages

        ``` bash
        pip install -r requirements.txt
        ```

## Use


## Method used

1. Use a checker board for calibration https://github.com/opencv/opencv/blob/4.x/doc/pattern.png

## Train your own classifier on your own dataset with google colab

Follow the link and follow the instructions

https://colab.research.google.com/drive/1qksJvIxpnAULe_8XfI04jG13a-3klOKA?usp=sharing


## Inspiration used to create this project

1. loading of images:

    https://ryanwingate.com/intro-to-machine-learning/deep-learning-with-pytorch/loading-image-data-into-pytorch/

2. https://docs.microsoft.com/en-us/learn/modules/intro-computer-vision-pytorch/2-image-data

3. Using a pretrained model to perform image classification

    https://github.com/LeanManager/PyTorch_Image_Classifier/blob/master/train.py

    Youtube video: https://www.youtube.com/watch?v=zFA8Cm13Xmk&t=513s