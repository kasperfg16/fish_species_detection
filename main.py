from ast import arg
from ctypes import sizeof
import shutil
from turtle import st
import cv2
import numpy as np
import os
import argparse
import predict
import rcnn_func as rcf
import functions_openCV as ftc
from functions_openCV import claheHSL
import precision_plot as pp
from os.path import exists

# Global variables
displayCorners = False
init_cali = True
init_load_model = True
resizePercent = 20

objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((6 * 9, 3), np.float32)
objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)


def save_coefficients(mtx, dist):

    '''Saves the coefficients from a camera calibration.'''

    global path

    """ Save the camera matrix and the distortion coefficients to given path/file. """
    cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
    cv_file.write("K", mtx)
    cv_file.write("D", dist)
    # note you *release* you don't close() a FileStorage object
    cv_file.release()


def load_coefficients():

    '''Loads the coefficients from a calibration file.'''

    basedir = os.path.dirname(os.path.abspath(__file__))
    cali_file = '/cali_matlab.yaml'
    path = basedir + cali_file

    # FILE_STORAGE_READ
    cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)

    camera_matrix = cv_file.getNode("K").mat()
    dist_matrix = cv_file.getNode("D").mat()

    cv_file.release()
    return camera_matrix, dist_matrix


def load_images_from_folder(folder):
    ''' Loads images from folder and returns an array. '''
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            imgG = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            images.append(imgG)

    return images


def resize_img(imgs, scale_percent, displayImages=False):
    """
    Resizes a list of images with a percentage.
    :param imgs: A list of images to resize
    :param scale_percent: The percent to scale the images by
    :return:
    """
    # If the parsed img is a list of images, then handle them as such and return a list. If not, just return the resized
    # image.
    if type(imgs) == list:
        resized_list = []
        for n in imgs:
            width = int(n.shape[1] * scale_percent / 100)
            height = int(n.shape[0] * scale_percent / 100)
            dim = (width, height)
            resized = cv2.resize(n, dim, interpolation=cv2.INTER_AREA)
            resized_list.append(resized)
            if displayImages:
                cv2.imshow("Undistorted", resized)
                cv2.waitKey(0)

        return resized_list

    else:
        width = int(imgs.shape[1] * scale_percent / 100)
        height = int(imgs.shape[0] * scale_percent / 100)
        dim = (width, height)
        resized = cv2.resize(imgs, dim, interpolation=cv2.INTER_AREA)

        return resized


def calibrate_camera():

    """
    Calibrates a camera using checkerboard pattern (6x9)

    :return: Saves camera parameters in an YAML file
    """

    global displayCorners

    imgs = load_images_from_folder("calibration_imgs")
    print(len(imgs))

    # Display image to see if the first image was read correctly and is grayscale
    cv2.imshow("Distorted", imgs[0])
    cv2.waitKey(0)

    print("Calibrating...")

    for img_g in imgs:

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(img_g, (9, 6), None)

        if ret:

            print("Found object points and image points, finding corners...")
            # Append the corners and object points
            objpoints.append(objp)
            imgpoints.append(corners)

            # Draw and display the corners
            if displayCorners:
                # Get closer corner estimation
                corners2 = cv2.cornerSubPix(img_g, corners, (11, 11), (-1, -1), criteria)
                # Draw the corners on the img
                cv2.drawChessboardCorners(img_g, (9, 6), corners2, ret)
                # Display the image
                cv2.imshow("Found corners", img_g)
                cv2.waitKey(0)

    print("Finding Camera Matrix and Determinants...")
    ret, mtx, dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, imgs[0].shape[::-1], None, None)
    save_coefficients(mtx, dist)
    print("Done calibrating!")
    cv2.destroyWindow("Image")

    print("Calibrated the cameras and saved values to cali.yaml")


def undistort_imgs(images, displayImages=False):

    """
    Undistorts images using mtx and dist derived from a YAML file.

    :param images: A list images to undistort
    :param displayImages: Display the undistorted images
    :return: A list of undistorted images
    """

    print("Started undistorting the images!")

    mtx, dist = load_coefficients()
    undi_img = []

    for n in images:
        height, width = n.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (width, height), 1, (width, height))

        # undistort
        dst = cv2.undistort(n, mtx, dist, None, newcameramtx)

        # Display image
        if displayImages:
            cv2.imshow("Undistorted", n)
            cv2.waitKey(0)

        # crop the image
        x, y, w, h = roi
        dst = dst[y:y + h, x:x + w]

        undi_img.append(dst)

    print("Done undistorting the images!")

    return undi_img


def distance_between_points(p1, p2):
    ''' Returns the distance between two points '''
    distance = ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5
    return abs(distance)


def get_contour_middle(cnt):
    M = cv2.moments(cnt)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        # set values as what you need in the situation
        cX, cY = 0, 0
    
    center = (cX, cY)

    return center


def isolate_fish(imgs, img_list_fish, display=False):
    """
    Isolates fish from a list of images.

    :param imgs: List of images of fish
    :param img_list_fish: ID of the fish
    :param display: Display each isolated image
    :return: A list of isolated fish images
    """

    # Apply CLAHE
    CLAHE = claheHSL(imgs, 2, (25, 25))

    # Threshold to create a mask for each image
    mask_cod, segmented_images = ftc.segment_codOPENCV(imgs)
    mask_cod_CLAHE, imgs_segmented_cod_CLAHE = ftc.segment_cod_CLAHEOPENCV(CLAHE)

    # CLAHE images
    ftc.save_imgOPENCV(
        imgs_segmented_cod_CLAHE,
        img_list_fish, path='fish_pics/output_images/',
        img_tag="_openCV_MANUAL_INSPECTION_CLAHE")

    # CLAHE images
    ftc.save_imgOPENCV(
        mask_cod, 
        img_list_fish, 
        path='fish_pics/output_images/manual_inspection_CLAHE', 
        img_tag="_openCV_MANUAL_INSPECTION_CLAHE")

    # Find the biggest contour
    count = 0
    isolated_fish = []
    conlistReturn = []
    bounding_boxes = []
    for n in imgs:
        biggestArea = 0
        distanceToMiddle = 100000
        closestCon = None
        biggestCon = None
        contMiddle = None
        closestConMiddle = 0
        biggestConList = []
        copy = n.copy()

        # Get the middle of the image
        (h, w) = copy.shape[:2]

        # Create a new mask for each image
        mask = np.zeros(copy.shape[:2], dtype=copy.dtype)

        # Find contours
        fishContours, __ = cv2.findContours(mask_cod_CLAHE[count], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Find biggest contour and the closest one to the middle of the image
        biggestConList = sorted(fishContours, key=cv2.contourArea, reverse= True)

        # Find the closest contour to the middle, only check the biggest contours
        for cont in biggestConList[:5]:
            contMiddle = get_contour_middle(cont)
            distance = distance_between_points((w//2, h//2), contMiddle)
            cv2.line(copy, contMiddle, (w // 2, h // 2), (0, 255, 0), thickness=2)
            if distance < distanceToMiddle:
                distanceToMiddle = distance
                closestCon = cont
                contMiddle = get_contour_middle(cont)
                closestConMiddle = contMiddle

        # Add the contour to a list for later use
        conlistReturn.append(closestCon)

        # Save the bounding box for later use in dataset
        x, y, w, h = cv2.boundingRect(closestCon)
        xMin, yMin, xMax, yMax = x, y, x + w, y + h
        boundBox = [xMin, yMin, xMax, yMax]
        bounding_boxes.append(boundBox)

        # Draw the contour and center of the shape on the image
        cv2.drawContours(copy, [closestCon], -1, (0, 255, 0), 2)
        cv2.circle(copy, closestConMiddle, 7, (255, 255, 255), -1)
        cv2.circle(copy, (w//2, h//2), 7, (255, 0, 255), -1)

        # Draw contours and isolate the fish
        cv2.drawContours(mask, [closestCon], 0, 255, -1)
        result = cv2.bitwise_and(n, n, mask=mask)
        isolated_fish.append(result)

        if display:
            cv2.imshow("Isolated fish", result)
            cv2.imshow("Normal fish - copy", copy)
            cv2.imshow("Normal fish - original", n)
            cv2.waitKey(0)

        count = count + 1

    return isolated_fish, conlistReturn, mask_cod, bounding_boxes


def parse_arguments():
    """
    Parses arguments for the program.

    :return: The arguments specified
    """
    parser = argparse.ArgumentParser(description='Image Classifier Predictions')

    # Command line arguments
    parser.add_argument('--image_dir', type=str, default="./fish_pics/input_images/", help='Absolute path to images')
    parser.add_argument('--checkpoint', type=str,
                        default='./checkpoint.pth',
                        help='Path to checkpoint')
    parser.add_argument('--image_dir_rcnn_images', type=str, default="./fish_pics/rcnn_masks/annotations/images/", help='Absolute path to image folder')
    parser.add_argument('--image_dir_rcnn_annotations', type=str, default="./fish_pics/rcnn_masks/annotations/annotations/", help='Absolute path to annotation folder')
    parser.add_argument('--train_rcnn', type=bool, default=True, help='Train mask rcnn classifier')
    parser.add_argument('--run_prediction_model', type=bool, default=False, help='Classify undistorted images')
    parser.add_argument('--topk', type=int, default=5, help='Top k classes and probabilities')
    parser.add_argument('--json', type=str, default='classes_dictonary.json', help='class_to_name json file')
    parser.add_argument('--device', type=str, default='cuda', help='\'cuda\' for GPU or \'cpu\' for CPU')
    parser.add_argument('--arUco_marker_cur', type=float, default=19.2, help='ArUco marker circumference')
    parser.add_argument('--calibrate_cam', type=bool, default=False, help='Set to \'True\' to re-calibrate camera. '
                        'Remember to put images of checkerboard in calibration_imgs folder')
    parser.add_argument('--undistorted', type=bool, default=False, help='Classify undistorted images')
    parser.add_argument('--make_new_data_set', type=bool, default=False, help='Use images in fish_pics\input_images and create a new dataset')
    parser.add_argument('--model_name', type=str, default='model_1', help='Select the model that we want to use for instance segmentation')

    arguments = parser.parse_args()

    return arguments


def load_ArUco_cali_objectsize_and_display(imgs, fish_names, fishContours, arguments, prediction):

    """
    Uses an ArUco marker to calibrate and predict the fish size.

    :param prediction: The predicted species
    :param imgs: The list of images used for prediction
    :param fishContours: All the contours o the fish
    :param arguments: The arguments for prediction
    :return: Displays fish size
    """

    len_estimate = []

    # Load ArUco image for calibration
    basedir = os.path.dirname(os.path.abspath(__file__))
    aruco_marker_img_name = '/arUco_in_box.JPG'
    aruco_marker_img_path = basedir + aruco_marker_img_name
    aruco_marker_img = cv2.imread(aruco_marker_img_path)

    # Undistort image
    list_aruco = [aruco_marker_img]
    aruco_marker_img_undi_list = undistort_imgs(list_aruco)
    aruco_marker_img = aruco_marker_img_undi_list[0]
    aruco_marker_img = resize_img(aruco_marker_img, resizePercent)
    cv2.imshow("aruco_marker_img", aruco_marker_img)
    cv2.waitKey(0)

    # Get parameters
    parameters = cv2.aruco.DetectorParameters_create()
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_50)

    # Detect corners of the ArUco Marker
    corners, _, _ = cv2.aruco.detectMarkers(aruco_marker_img, aruco_dict, parameters=parameters)

    # Check if any marker were detected
    if not corners:
        print("No arUco markers found in image.")
        exit()
    elif corners:
        int_corners = np.int0(corners)
        cv2.polylines(aruco_marker_img, int_corners, True, (0, 255, 0), 5)

        # ArUco parameters
        aruco_parameters = cv2.arcLength(corners[0], True)

        # Pixel to cm ratio
        pixel_cm_ratio = aruco_parameters / arguments.arUco_marker_cur

        print("Camera setup calibrated")

    # Display all the details of each fish in each image
    count = 0
    for n in imgs:
        rect = cv2.minAreaRect(fishContours[count])
        (x, y), (w, h), angle = rect

        # Get width and height of objects in cm
        w_cm = round(w / pixel_cm_ratio, 2)
        h_cm = round(h / pixel_cm_ratio, 2)

        # Because of some rotation issues doing width and height calculations, we need to change up the width and
        # height sometimes to make sure the correct values are set for each variable. Since we know the fish will always
        # have a greater width than height, we can simply do a check and then change up the variables if the check is
        # true.
        if h_cm > w_cm:
            h_cm, w_cm = w_cm, h_cm

        box = cv2.boxPoints(rect)
        box = np.int0(box)

        # Display for the user
        cv2.circle(n, (int(x), int(y)), 5, (0, 0, 255), -1)

        cv2.polylines(n, [box], True, (255, 0, 0), 2)

        cv2.putText(n, "Width {} cm".format(w_cm, 1), (int(x - 300), int(y - 80)), cv2.FONT_HERSHEY_PLAIN, 2,
                    (100, 200, 0), 2)
        cv2.putText(n, "Height {} cm".format(h_cm, 1), (int(x + 0), int(y - 80)), cv2.FONT_HERSHEY_PLAIN, 2,
                    (100, 200, 0), 2)
        cv2.putText(n, "Species: {}".format(prediction[count], 1), (int(x - 100), int(y + 90)), cv2.FONT_HERSHEY_PLAIN, 2,
                    (100, 200, 0), 2)

        cv2.imshow("Picture: " + str(fish_names[count]), n)
        cv2.waitKey(0)
        cv2.destroyWindow("Picture: " + str(fish_names[count]))

        len_estimate.append(w_cm)
        
        count += 1

    return len_estimate


def create_dataset(arguments, imgs, fish_names, fish_masks, bounding_boxes, label, path_images, path_annotation):

    print("Creating dataset...")

    # Save normalized masks images in a folder
    normalized_masks = rcf.normalize_masks(fish_masks)
    counter = 0
    for mask in normalized_masks:
        cv2.imwrite(path_images + fish_names[counter] + ".png", mask)
        counter += 1
    
    rcf.save_annotations(imgs, bounding_boxes, fish_names, label, path_annotation)

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
    normalized_masks = rcf.normalize_masks(fish_masks)
    for mask in normalized_masks:
        path_img = os.path.join(path_masks, fish_names[counter])
        path_img = path_img + '.png'
        print(path_img)
        cv2.imwrite(path_img, mask)
        counter += 1

    rcf.save_annotations(imgs, bounding_boxes, fish_names, label, path_annotations)


def create_dataset_mask_rcnn(arguments):

    load_folder = '/fish_pics/input_images'

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
                dst = undistort_imgs(imgs)

                # Isolate fish contours
                isolatedFish, contoursFish, masks, bounding_boxes = isolate_fish(dst, img_list_fish, display=False)

                isolatedFish = resize_img(isolatedFish, 10)
                imgs = resize_img(dst, 10)

                # Create paths for folders
                path_dataset = '/fish_pics/rcnn_dataset/images'

                imgs_folder = '/fish_pics/rcnn_dataset/images'
                path_imgs_folder = basedir + imgs_folder

                masks_folder = '/fish_pics/rcnn_dataset/masks'
                path_masks_folder = basedir + masks_folder

                annotations_folder = '/fish_pics/rcnn_dataset/annotations'
                path_annotations_folder = basedir + annotations_folder

                # Delete old dataset if it exists 
                if exists(path_imgs_folder):
                    shutil.rmtree(path_dataset)
                
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

        rcf.validate_masks("fish_pics/rcnn_dataset/masks/")

    print("Done creating dataset!")


def main(args=None):

    # Load arguments
    arguments = parse_arguments()

    # Create dataset if requested
    if arguments.make_new_data_set:
        create_dataset_mask_rcnn(arguments)

    # Create path to model
    basedir = os.path.dirname(os.path.abspath(__file__))
    models_path = os.path.join(basedir, 'models')
    model_name = arguments.model_name
    model_path = os.path.join(models_path, model_name)
    
    # Check if we want to run the RCNN trainer
    if arguments.train_rcnn:
        # Run the RCNN trainer
        rcf.run_rcnn_trainer(model_path)
    
    imgs = []

    for img in imgs:
        rcf.predict_rcnn(img, model_path)

    # ArUco marker calibration for size estimation, displays results of the calculated size
    len_estimate = load_ArUco_cali_objectsize_and_display(isolatedFish, img_list_fish, contoursFish, arguments, predictions)

    # Precision calculation
    pp.calc_len_est(img_list_abs_path, len_estimate)

if __name__ == '__main__':
    main()
