# Fish_species_detection
Repo for fish species detection project at Aalborg University

## This project is best used with
https://colab.research.google.com/drive/1qksJvIxpnAULe_8XfI04jG13a-3klOKA?usp=sharing

## Setup

1. Open a terminal

    ### Create a conda enviroment

    Create a Conda environment using:

    ``` bash
    conda create -n env_pytorch python=3.6
    ```

    Activate the environment using:

    ``` bash
    conda activate env_pytorch
    ```

    ### Install pytorch

    Install PyTorch using conda:

    ``` bash
    conda install pytorch torchvision -c pytorch
    ```

    ### Install other dependancies

    ``` bash
    pip install pandas
    ```

## Inspiration

1. loading of images:

    https://ryanwingate.com/intro-to-machine-learning/deep-learning-with-pytorch/loading-image-data-into-pytorch/

2. https://docs.microsoft.com/en-us/learn/modules/intro-computer-vision-pytorch/2-image-data

3. Using a pretrained model to perform image classification

    https://github.com/LeanManager/PyTorch_Image_Classifier/blob/master/train.py

    Youtube video: https://www.youtube.com/watch?v=zFA8Cm13Xmk&t=513s