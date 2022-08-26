# Fish_species_detection - mask-rcnn-cleaned

Repo for fish species detection project at Aalborg University

## Methods used

1.
    A dataset of 156 sea creatures including 34 different species where made using a custom made photobox.
    - For each sea creature the dataset contains:
        - The lenght of the sea creature
        - The species of the sea creature
        - Two images from different sides of the sea creature

2.
    A calibration session is done where a checker board is used for calibration <https://github.com/opencv/opencv/blob/4.x/doc/pattern.png>

    The camera intrinsics and distortion coefficients are obtained in this session and saved.

3.
    The camera intrinsics and distortion coefficients are loaded and used to undistort the images of cods. The undistorted images are added to the dataset.

4.
    The fish is segmented using hue thresholding using the undistorted images. A mask from the segmentation is used saved and added to the dataset (This could also have been done using manual method with tools such as <https://www.makesense.ai/>).

5.
    We train an instance segmentation model on the custom dataset using the undistorted images and the masks of instances of cod in the undistorted images. This is done using mask-rcnn trhough pytorch. We use tranfer learning by taking pre-trained weights and fine tune parameters in the model using our custom dataset.

    The input to the model is an image and the output of the model is masks of istances of cod in the images.

    The general tutorial used is here: <https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html>.

    More on the pre-trained model and used weights here: <https://pytorch.org/vision/stable/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn_v2.html#torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights>.

6.
    An ArUco marker is printed and mounted on a flat board and put into the fish box so that the ArUco marker is roughly in the same plane as the fish: ![ArUco in box](arUco_in_box.JPG)

    This image is used with cv2 functions to find the circumference of the ArUco markers in pixels. The circumference is then measured with physically in mm. By deviding the circumference in pixel with the circumference in mm gives a mm/pixel ratio.

7.
    We now use the trained model to predict masks using undistorted test images.

8.
    A bounding box is created around the of cod in the masks images. The lenght of the fish is found by taking the mm/pixel ratio and applying to the lenght of the bounding box. This gives the lenght of the fish.

9.
    A sample of 145 cods is taken. The lenght estimation method is used and comparred with the lenght that were measured physically when the dataset was created.

    There is correlation between the measured and estimated data. Therefore the mean of the errors of the sample is subrated from the estimated lenght. This gives the following results:

    - <https://aaudk.sharepoint.com/:i:/r/sites/Fiskeprojekt/Delte%20dokumenter/General/Figure/Error_at_different_lenghts_avg_subtracted_mask_rcnn.png?csf=1&web=1&e=80Qjfn>

    - <https://aaudk.sharepoint.com/:i:/r/sites/Fiskeprojekt/Delte%20dokumenter/General/Figure/Real_vs_estimated_avg_subtracted_sample_size_312_mask_rcnn.png?csf=1&web=1&e=5IBTqD>

## Perquisites

- This is developed for windows and only tested on Window 10 and 11 OS

## Setup

1. Download the repository:

    ``` bash
    https://github.com/kasperfg16/fish_species_detection.git
    ```

2. Install conda

    ``` bash
    start /B /WAIT %UserProfile%\miniconda.exe /InstallationType=JustMe /AddToPath=0 /RegisterPython=0 /S /D=%UserProfile%\miniconda3
    ```

3. Open command prompt

    - a)

        Navigate to the repository folder (...**\fish_species_detection**)

        Example:

        ``` bash
        cd C:\Users\Kaspe\OneDrive\Onenote\GitHub\fish_species_detection
        ```

    - b)

        Create a conda environment

        ``` bash
        conda create -n env_pytorch python=3.10.4
        ```

    - c)

        Activate the environment:

        ``` bash
        conda activate env_pytorch
        ```

        Install required packages:

        ``` bash
        pip install -r requirements.txt
        ```

    Now you're ready to run the code

## How to use

1. ### Folder structure

    Create a folder structure as followed:

    - Project_Folder
        - fish_pics
            - input_images
                - cods
                    - (All the images that needs to be converted to a dataset for R-CNN)
        - rcnn_dataset
            - annotations (Currently only used for saving each photos annotation)
            - images
                - (All the images from the dataset creator will be saved here)
            - masks
                - (All the masks from the dataset creator will be saved here)
            - validation
                - (The output from the model-guesses will be saved in this folder)
        - models
            - model_1 (for now, the model needs to be named "model_1" to be found. This will be changed in the future.)

2. ### See all the arguments

    E.g. for [train_k_fold_val.py](train_k_fold_val.py) run:

    ``` bash
    python train_k_fold_val.py --help
    ```

3. ### Calibrate camera

    Take images of checkerboard pattern (6x9) with the camera that you want to calibrate and save images in [calibration_imgs](calibration_imgs) folder. [Checkerboard pattern (6x9)](https://github.com/opencv/opencv/blob/4.x/doc/pattern.png)

    When running [train_k_fold_val.py](train_k_fold_val.py) use the undistort argument like so: `python .\train_k_fold_val.py --calibrate_cam True`

    The same can be done with `python .\main.py --calibrate_cam True`

4. ### Train the model

    To train the model put images into [fish_pics\input_images](fish_pics\input_images)

    To add another class, create a new folder inside [fish_pics\input_images](fish_pics\input_images) and upload images of this new class into this folder. E.g **fish_pics\input_images\tortiose**

    Then run [train_k_fold_val.py](train_k_fold_val.py) like so: `python .\main.py --num_of_k 1`

    To stop the training press ctrl+c **ONCE**. This will stop the training at the point where it is and save the model.

5. ### Run main code

    Run [main.py](main.py)

## Train your own R-CNN on your own dataset with google colab

Follow the link and follow the instructions

[Google Colab R-CNN masking](https://colab.research.google.com/drive/1oxojIhiJwssvCTv5AOTZ3y7nDfHh2IfN?usp=sharing)

## Inspiration used to create this project

1. loading of images:

    <https://ryanwingate.com/intro-to-machine-learning/deep-learning-with-pytorch/loading-image-data-into-pytorch/>

2. <https://docs.microsoft.com/en-us/learn/modules/intro-computer-vision-pytorch/2-image-data>

3. Using a pretrained model to perform image classification

    <https://github.com/LeanManager/PyTorch_Image_Classifier/blob/master/train.py>

    Youtube video: <https://www.youtube.com/watch?v=zFA8Cm13Xmk&t=513s>

4. For R-CNN masking a PyTorch tutorial was followed and implemented. The dataset is created to follow the same structure as PennFudan dataset which is presented in the tutorial.

    <https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html>
