import cv2
import numpy as np
import os
import argparse
import predict
import functions_openCV as ftc
from functions_openCV import claheHSL

# Global variables
path = os.path.abspath("cali_matlab.yaml")
# path = os.path.abspath("cali_gopro_opencv.yaml")
image_test_undis = os.path.abspath("fish_pics/input_images/cods/undis.jpg")

cali = False
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

    global path

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


def resize_img(imgs, scale_percent):
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

    imgs = load_images_from_folder("Calibration")
    print(len(imgs))

    # Display image to see if the first image was read correctly and is grayscale
    cv2.imshow("Undistorted", imgs[0])
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

    return undi_img


def distance_between_points(p1, p2):
    ''' Returns the distance between two points '''
    distance = ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5
    return abs(distance)


def get_contour_middle(cnt):
    M = cv2.moments(cnt)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
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
    mask_cod_CLAHE, img_segmented_cod_CLAHE = ftc.segment_cod_CLAHEOPENCV(CLAHE)

    # CLAHE images
    ftc.save_imgOPENCV(img_segmented_cod_CLAHE, 'fish_pics/output_images/', img_list_fish,
                       "_openCV_MANUAL_INSPECTION_CLAHE")

    # CLAHE images
    ftc.save_imgOPENCV(mask_cod, 'fish_pics/output_images/manual_inspection_CLAHE', img_list_fish,
                       "_openCV_MANUAL_INSPECTION_CLAHE")

    # Find the biggest contour
    count = 0
    isolated_fish = []
    conlistReturn = []
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
        for cont in fishContours:
            area = cv2.contourArea(cont)
            if area > biggestArea:
                biggestArea = area
                biggestCon = cont
                biggestConList.append(biggestCon)

        # Remove the background contour, as it is always the biggest contour in the image
        del biggestConList[-1]

        # Find the closest contour to the middle, only check the biggest contours
        for cont in biggestConList:
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

    return isolated_fish, conlistReturn


def parse_arguments():
    """
    Parses arguments for the program.

    :return: The arguments specified
    """
    parser = argparse.ArgumentParser(description='Image Classifier Predictions')

    # Command line arguments
    parser.add_argument('--image_dir', type=str, default="./fish_pics/input_images/", help='Absolute path to images')
    parser.add_argument('--image_dir_cods', type=str, default="./fish_pics/input_images/cods/", help='Absolute path to images')
    parser.add_argument('--checkpoint', type=str,
                        default='./checkpoint.pth',
                        help='Path to checkpoint')
    parser.add_argument('--topk', type=int, default=5, help='Top k classes and probabilities')
    parser.add_argument('--json', type=str, default='classes_dictonary.json', help='class_to_name json file')
    parser.add_argument('--gpu', type=str, default='cuda', help='GPU or CPU')
    parser.add_argument('--arUco_marker_cur', type=float, default=19.2, help='ArUco marker circumference')

    arguments = parser.parse_args()

    return arguments


def load_predict_model(img_list, arguments):
    """
    Loads the prediction model for prediction.

    :param arguments: The arguments for the predict model
    :return: The predicted species
    """
    # Load model (only needed once at startup)
    print("Loading model...")
    checkpoint, model, class_to_name_dict, device = predict.load_predition_model(arguments.checkpoint, arguments.image_dir)
    print("Model loaded")

    # Predict
    predictions = predict.predict_species(img_list, arguments.topk, checkpoint, model, class_to_name_dict,
                                         device)

    return predictions


def load_ArUco_cali_objectsize_and_display(imgs, fishContours, arguments, prediction):

    """
    Uses an ArUco marker to calibrate and predict the fish size.

    :param prediction: The predicted species
    :param imgs: The list of images used for prediction
    :param fishContours: All the contours of the fish
    :param arguments: The arguments for prediction
    :return: Displays fish size
    """

    # Load ArUco image for calibration
    aruco_marker_img = cv2.imread(os.path.abspath("arUco_in_box.JPG"))
    list_aruco = [aruco_marker_img]
    aruco_marker_img_undi_list = undistort_imgs(list_aruco)
    aruco_marker_img = aruco_marker_img_undi_list[0]
    aruco_marker_img = resize_img(aruco_marker_img, resizePercent)

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

        cv2.imshow("size", n)
        cv2.waitKey(0)

        count += 1


def main(args=None):

    # Load arguments
    arguments = parse_arguments()

    global image_test_undis

    # Load all the images
    images, img_list_fish, img_list_abs_path = ftc.loadImages(arguments.image_dir_cods, edit_images=False, show_img=False)

    # Do we want to calibrate before undistorting the image?
    if cali:
        calibrate_camera()
    else:
        # Undistorts images
        dst = undistort_imgs(images)

        # Resize images for presentation
        resized = resize_img(dst, resizePercent)

        # Isolate fish contours
        isolatedFish, contoursFish = isolate_fish(resized, img_list_fish, display=False)

        # Load and predict using the model
        predictions = load_predict_model(img_list_abs_path, arguments)

        # ArUco marker calibration for size estimation, displays results of the calculated size
        load_ArUco_cali_objectsize_and_display(isolatedFish, contoursFish, arguments, predictions)


if __name__ == '__main__':
    main()
