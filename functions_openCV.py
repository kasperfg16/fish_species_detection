import copy
import os
import cv2
import camera_cal as ccal
import functions as func
import numpy as np
from matplotlib import pyplot as plt


def resize_imgs(imgs, scale_percent, displayImages=False):
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


def segment_codOPENCV(images, show_images=False):
    print("Started segmenting the cod!")

    inRangeImages = []
    segmentedImages = []

    for n in images:
        hsv_img = copy.copy(n)
        hsv_img = cv2.cvtColor(hsv_img, cv2.COLOR_BGR2HSV)

        # Create threshold for segmenting cod
        mask = cv2.inRange(hsv_img, (100, 21, 65), (180, 255, 255))

        # Invert the mask
        mask = (255 - mask)

        # Create kernels for morphology
        # kernelOpen = np.ones((4, 4), np.uint8)
        # kernelClose = np.ones((7, 7), np.uint8)

        kernelOpen = np.ones((3, 3), np.uint8)
        kernelClose = np.ones((5, 5), np.uint8)

        # Perform morphology
        open1 = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernelOpen, iterations=3)
        close2 = cv2.morphologyEx(open1, cv2.MORPH_CLOSE, kernelClose, iterations=5)

        segmented_cods = cv2.bitwise_and(n, n, mask=close2)

        segmented_cods[close2 == 0] = (255, 255, 255)

        if show_images:
            cv2.imshow("res", segmented_cods)
            cv2.imshow("mask", mask)
            cv2.waitKey(0)

        # add to lists
        inRangeImages.append(mask)
        segmentedImages.append(segmented_cods)

    print("Finished segmenting the cod!")

    return inRangeImages, segmentedImages

def save_imgOPENCV(imgs, originPathNameList, path='', img_tag=''):
    '''
    Saves a list of images in the folder that the path is set to.

    :param originPathName: The path of the original path of the images that have been manipulated.
    :param imgs: A list of images.
    :param path: The path that the images will be saved to.
    :param img_tag: Tag that is is added to the original image file name e.g. filname + img_tag = filnameimg_tag.JPG
    :param numbering: Set to true if the image names should be numbered when saved
    :return: None
    '''

    print('Saving images in:', path)

    count = 0
    if len(imgs) > 0:
        for n in imgs:
            cv2.imwrite(path + f"\\{originPathNameList[count] + img_tag}.JPG", n)
            count = count + 1


    print('Done saving images')


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
    mask_cod, segmented_images = segment_codOPENCV(imgs)
    mask_cod_CLAHE, imgs_segmented_cod_CLAHE = segment_cod_CLAHEOPENCV(CLAHE)

    # CLAHE images
    save_imgOPENCV(
        imgs_segmented_cod_CLAHE,
        img_list_fish, path='fish_pics/output_images/',
        img_tag="_openCV_MANUAL_INSPECTION_CLAHE")

    # CLAHE images
    save_imgOPENCV(
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
            distance = func.distance_between_points((w//2, h//2), contMiddle)
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


def load_ArUco_cali_objectsize_and_display(imgs, fish_names, fishContours, arguments, prediction, display=False):

    """
    Uses an ArUco marker to calibrate and predict the fish size.

    :param prediction: The predicted species
    :param imgs: The list of images used for prediction
    :param fishContours: All the contours o the fish
    :param arguments: The arguments for prediction
    :return: Displays fish size
    """

    print("Started loading arUco calibration for object size estimation...")

    len_estimate = []

    # Load ArUco image for calibration
    basedir = os.path.dirname(os.path.abspath(__file__))
    aruco_marker_img_name = '/arUco_in_box.JPG'
    aruco_marker_img_path = basedir + aruco_marker_img_name
    aruco_marker_img = cv2.imread(aruco_marker_img_path)

    # Undistort image
    list_aruco = [aruco_marker_img]
    aruco_marker_img_undi_list = ccal.undistort_imgs(list_aruco)
    aruco_marker_img = aruco_marker_img_undi_list[0]
    aruco_marker_img = resize_imgs(aruco_marker_img, 10)
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
    fish_names_sorted = []
    for n in imgs:
        
        # Since we are using minAreaRect to get the fish size, we need to convert the contours to a list of points
        # https://stackoverflow.com/questions/71990194/opencv-minarearect-points-is-not-a-numerical-tuple
        contours_flat = np.vstack(fishContours[count]).squeeze()
        rect = cv2.minAreaRect(contours_flat)

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
        if display:
            cv2.circle(n, (int(x), int(y)), 5, (0, 0, 255), -1)

            cv2.polylines(n, [box], True, (255, 0, 0), 2)

            cv2.putText(n, "Width {} cm".format(w_cm, 1), (int(x - 100), int(y - 80)), cv2.FONT_HERSHEY_PLAIN, 1.1,
                        (100, 200, 0), 2)
            cv2.putText(n, "Height {} cm".format(h_cm, 1), (int(x + 0), int(y - 50)), cv2.FONT_HERSHEY_PLAIN, 1.1,
                        (100, 200, 0), 2)
            #cv2.putText(n, "Species: {}".format(prediction[count], 1), (int(x - 100), int(y + 90)), cv2.FONT_HERSHEY_PLAIN, 1.1,
                        #(100, 200, 0), 2)
            cv2.putText(n, "Species: {}".format(prediction, 1), (int(x - 100), int(y + 90)), cv2.FONT_HERSHEY_PLAIN, 1.1,
                        (100, 200, 0), 2)

            cv2.imshow("Picture: " + str(fish_names[count]), n)
            cv2.waitKey(0)
        # cv2.destroyWindow("Picture: " + str(fish_names[count]))

        # Save the ordered names so they match the order of the length estimates
        fish_names_sorted.append(fish_names[count])
        len_estimate.append(w_cm)
        
        count += 1

    return len_estimate, fish_names_sorted


def segment_cod_CLAHEOPENCV(images, show_images=False):
    print("Started segmenting the cod!")

    inRangeImages = []
    segmentedImages = []

    for n in images:
        hsv_img = copy.copy(n)
        hsv_img = cv2.cvtColor(hsv_img, cv2.COLOR_BGR2HSV)

        # Create threshold for segmenting cod
        mask = cv2.inRange(hsv_img, (99, 15, 30), (123, 255, 255))

        # Invert the mask
        mask = (255 - mask)

        # Create kernels for morphology
        # kernelOpen = np.ones((4, 4), np.uint8)
        # kernelClose = np.ones((7, 7), np.uint8)

        kernelOpen = np.ones((3, 3), np.uint8)
        kernelClose = np.ones((5, 5), np.uint8)

        # Perform morphology
        open1 = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernelOpen, iterations=3)
        close2 = cv2.morphologyEx(open1, cv2.MORPH_CLOSE, kernelClose, iterations=5)

        segmented_cods = cv2.bitwise_and(n, n, mask=close2)

        segmented_cods[close2 == 0] = (255, 255, 255)

        if show_images:
            cv2.imshow("res", segmented_cods)
            cv2.imshow("mask", mask)
            cv2.waitKey(0)

        # add to lists
        inRangeImages.append(mask)
        segmentedImages.append(segmented_cods)

    print("Finished segmenting the cod!")

    return inRangeImages, segmentedImages


def claheHSL(imgList, clipLimit, tileGridSize):
    '''
    Performs CLAHE on a list of images
    '''
    fiskClaheList = []
    for img in imgList:
        fiskHLS2 = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        LChannelHLS = fiskHLS2[:, :, 1]
        clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
        claheLchannel1 = clahe.apply(LChannelHLS)
        fiskHLS2[:, :, 1] = claheLchannel1
        fiskClahe = cv2.cvtColor(fiskHLS2, cv2.COLOR_HLS2BGR)
        fiskClaheList.append(fiskClahe)
    return fiskClaheList