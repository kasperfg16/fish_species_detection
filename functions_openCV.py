import copy
import os
import cv2
import camera_cal as ccal
import functions as func
import numpy as np
from matplotlib import pyplot as plt


def checkerboard_calibrateOPENCV(dimensions, images_distort, images_checkerboard, show_img=False, recalibrate=False):
    """
    Undistorts images by a checkerboard calibration.

    SRC: https://docs.opencv.org/master/dc/dbb/tutorial_py_calibration.html

    :param show_img: Debug to see if all the images are loaded and all the edges are found
    :param dimensions: The dimensions of the checkerboard from a YAML file
    :param images_distort: The images the needs to be undistorted
    :param images_checkerboard: The images of the checkerboard to calibrate by
    :return: If it succeeds, returns the undistorted images, if it fails, returns the distorted images with a warning
    """
    print('Undistorting images ... \n')

    if recalibrate:
        print('Calibrating camera please wait ... \n')

        chessboardSize = (6, 9)

        # termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 25, 0.001)

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((dimensions[0][1] * dimensions[1][1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:6, 0:9].T.reshape(-1, 2)

        # Arrays to store object points and image points from all the images.
        objpoints = []  # 3d point in real world space
        imgpoints = []  # 2d points in image plane.

        for img in images_checkerboard:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, chessboardSize, None)

            # If found, add object points, image points (after refining them)
            if ret is True:
                objpoints.append(objp)
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners)
                # Draw and display the corners
                cv2.drawChessboardCorners(img, chessboardSize, corners2, ret)
                if show_img:
                    print(imgpoints)
                    cv2.imshow('img', img)
                    cv2.imshow('gray', gray)
                    cv2.waitKey(0)

        # Calibrate the camera
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[:2], None, None)

        print("Done calibrating")

        # Save calibration session parameters
        print('Saving calibration session parameters in calibration/parameters_calibration_session ... \n')
        np.save('calibration/parameters_calibration_session/mtx.npy', mtx)
        np.save('calibration/parameters_calibration_session/dist.npy', dist)

    # Loading in parameters from previous calibration session
    mtx = np.load('calibration/parameters_calibration_session/mtx.npy')
    dist = np.load('calibration/parameters_calibration_session/dist.npy')

    print("Intrinsic parameters:")
    print("Camera matrix: K =")
    print(mtx)
    print("\nDistortion coefficients =")
    print(dist, "\n")

    # Go through all the images and undistort them
    img_undst = []
    for n in images_distort:
        # Get image shape
        h, w = n.shape[:2]

        # Refine the camera matrix
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

        # Undistort using remapping
        mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w, h), 5)
        undst = cv2.remap(n, mapx, mapy, cv2.INTER_LINEAR)

        img_undst.append(undst)
        if show_img:
            cv2.imshow('calibresult.png', undst)
            cv2.waitKey(0)

    print("Done undistorting images")

    return img_undst


def detect_woundspotsOPENCV(imgs, maskCod):
    '''
    Detect bloodspots, mark and tag them and find the coverage of bloodspots on hte cod

    :param imgs: Images with cod
    :param maskCod: The mask showing only the cod area
    :return: mask of blood spots, segmented blood spots, marked and tagged blood spots, coverage of blood spots on the
    cod
    '''
    mask_woundspots = []
    segmented_blodspots_imgs = []
    marked_woundspots_imgs = []
    percSpotCoverage = []
    count = 0

    # Find biggest contour
    for n in imgs:
        hsv_img = cv2.cvtColor(copy.copy(n), cv2.COLOR_BGR2HSV)
        biggestarea = 0
        fishContours, __ = cv2.findContours(maskCod[count], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for cont in fishContours:
            area = cv2.contourArea(cont)
            if area > biggestarea:
                biggestarea = area

        fishArea = biggestarea

        spotcount = 0

        marked_woundspots_imgs.append(copy.copy(n))

        # Threshold for blood spots
        frame_threshold1 = cv2.inRange(hsv_img, (0, 90, 90), (10, 255, 255))

        # Combining the masks
        mask_woundspots.append(frame_threshold1)

        # Create kernels for morphology
        kernelClose = np.ones((30, 30), np.uint8)

        # Perform morphology
        close = cv2.morphologyEx(mask_woundspots[count], cv2.MORPH_CLOSE, kernelClose)

        # Perform bitwise operation to show woundspots instead of BLOBS
        segmented_blodspots_imgs.append(cv2.bitwise_and(n, n, mask=close))

        # Make representation of BLOB / woundspots
        # Find contours
        contours, _ = cv2.findContours(close, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Classify as blood spots if the spots are big enought
        totalSpotArea = 0
        for cont in contours:
            area = cv2.contourArea(cont)
            if area > 50:
                x, y, w, h = cv2.boundingRect(cont)
                # Create tag
                cv2.putText(marked_woundspots_imgs[count], 'Wound', (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255),
                            3)
                # Draw green contour
                cv2.rectangle(marked_woundspots_imgs[count], (x - 5, y - 5), (x + w + 5, y + h + 5), (0, 255, 0), 2);

                # Because the biologists have put a red tag on the cod, there will always be detected at least 1 blood
                # spot
                spotcount = spotcount + 1
                if spotcount > 1:
                    totalSpotArea = totalSpotArea + area

        percSpotCoverage.append(totalSpotArea / fishArea * 100)

        count = count + 1

    return mask_woundspots, marked_woundspots_imgs, segmented_blodspots_imgs, percSpotCoverage


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


def resize_img_single(img, scale_percent):
    """
    Resizes the image by a scalar

    :param img: the image to resize
    :param scale_percent: The scaling percentage
    :return: the image scaled by the scalar
    """
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)

    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    # resize image
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


def showSteps(stepsList, CLAHE=False):
    '''
    Create subplots showing main steps in algorithm

    :return: None
    '''

    # OpenCV loads pictures in BGR, but the this step is plotted in RGB:
    count = 0
    img_rgb = cv2.cvtColor(stepsList[count], cv2.COLOR_BGR2RGB)
    count = count + 1
    img_undistorted_rgb = cv2.cvtColor(stepsList[count], cv2.COLOR_BGR2RGB)
    count = count + 1
    img_cropped_rgb = cv2.cvtColor(stepsList[count], cv2.COLOR_BGR2RGB)
    count = count + 1
    if CLAHE:
        cod_CLAHErgb = cv2.cvtColor(stepsList[count], cv2.COLOR_BGR2RGB)
        count = count + 1
    img_segmented_codrgb = cv2.cvtColor(stepsList[count], cv2.COLOR_BGR2RGB)
    count = count + 1
    bloodspotsrgb = cv2.cvtColor(stepsList[count], cv2.COLOR_BGR2RGB)
    count = count + 1
    marked_bloodspotssrgb = cv2.cvtColor(stepsList[count], cv2.COLOR_BGR2RGB)

    fig = plt.figure()
    fig.suptitle('Steps in algorithm', fontsize=16)

    subplot = 1
    plt.subplot(3, 3, subplot)
    plt.imshow(img_rgb)
    plt.title('Original image')
    subplot = subplot + 1

    plt.subplot(3, 3, subplot)
    plt.imshow(img_undistorted_rgb)
    plt.title('Undistorted image')
    subplot = subplot + 1

    plt.subplot(3, 3, subplot)
    plt.imshow(img_cropped_rgb)
    plt.title('ROI')
    subplot = subplot + 1

    if CLAHE:
        plt.subplot(3, 3, subplot)
        plt.imshow(cod_CLAHErgb)
        plt.title('CLAHE')
        subplot = subplot + 1

    plt.subplot(3, 3, subplot)
    plt.imshow(img_segmented_codrgb)
    plt.title('Segmented cod')
    subplot = subplot + 1

    plt.subplot(3, 3, subplot)
    plt.imshow(bloodspotsrgb)
    plt.title('Blood spots segmented')
    subplot = subplot + 1

    plt.subplot(3, 3, subplot)
    plt.imshow(marked_bloodspotssrgb)
    plt.title('Blood spots tagged')
    subplot = subplot + 1

    plt.show()


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


def crop(images, y, x, height, width):
    '''
    Crops images.

    :param images: The images to crop
    :return: Cropped images
    '''

    print("Cropping images ... ")
    cropped_images = []
    for n in images:
        ROI = n[y:y + height, x:x + width]
        cropped_images.append(ROI)

    print("Done cropping images!")

    return cropped_images


def loadImages(folder='/fish_pics/input_images/cods/', edit_images=False, show_img=False, scaling_percentage=30, full_path=False):
    """
        Loads all the images inside a file.

        :return: All the images in a list and its file names.
    """

    print("Loading the images!")

    images = []
    img_names = []
    img_list_abs_path = []

    # Find path to image folder
    if not full_path:
        path_img_folder = folder
        basedir = os.path.dirname(os.path.abspath(__file__))
        path = basedir + path_img_folder
    else:
        path = folder

    # Create list of img paths
    img_list = os.listdir(path)
    print("Loading in images from:", path, "...")
    print("Total images found:", len(img_list))
    
    
    for cl in img_list:
        # Find all the images in the file and save them in a list without the ".jpg"
        cur_img = cv2.imread(f"{path}/{cl}", 1)
        img_name = os.path.splitext(cl)[0]

        # Do some quick images processing to get better pictures if the user wants to
        if edit_images:
            cur_img_re = resize_img_single(cur_img, scaling_percentage)
            cur_img = cur_img_re

        # Show the image before we append it, to make sure it is read correctly
        if show_img:
            cv2.imshow(f"Loaded image: {img_name}", cur_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # Append them into the list
        images.append(cur_img)
        img_names.append(img_name)
        img_list_abs_path.append(path+cl)

    # Remove the image window after we have checked all the pictures
    cv2.destroyAllWindows()

    print("Done loading the images!")

    return images, img_names, img_list_abs_path


def load_images_from_folder_grayscale(folder):
    ''' Loads images from folder and returns an array. '''
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            imgG = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            images.append(imgG)

    return images


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


def load_ArUco_cali_objectsize_and_display(imgs, fish_names, fishContours, arguments, prediction, precisions, display=False):

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
            cv2.putText(n, "Type: {}".format(prediction[count], 1), (int(x - 100), int(y + 90)), cv2.FONT_HERSHEY_PLAIN, 1.1,
                        (100, 200, 0), 2)
            cv2.putText(n, "Precision: {}".format(precisions[count], 1), (int(x), int(y + 90)), cv2.FONT_HERSHEY_PLAIN, 1.1,
                        (100, 200, 0), 2)
            cv2.imshow("Picture: " + str(fish_names[count]), n)
            cv2.waitKey(0)

        # Save the ordered names so they match the order of the length estimates
        fish_names_sorted.append(fish_names[count])
        len_estimate.append(w_cm)
        
        count += 1

    return len_estimate, fish_names_sorted


def saveCDI(img_list_fish, percSpotCoverage):
    """
    Saves a CDI of each fish in a .txt file.

    :param img_list_fish: The names of each fish
    :param percSpotCoverage: The percentage the wound covers the surface area of one side of the fish
    :return:
    """

    print("Started CDI...")

    f = open("output_CDI/CDI.txt", "w+")

    line = ("-----------------------------------------------------------------\n")

    # Layout for CDI
    f.write("CDI\n\n")
    f.write("              |Category: \t|Wounds \n")
    f.write(line)
    f.write("FISH:\n")

    # Fill CDI
    for i in range(len(img_list_fish)):
        name = os.path.splitext(img_list_fish[i])[0]

        if percSpotCoverage[i] > 0:
            f.write(line)
            f.write("%s \t\t\t|x, coverage = " % name)
            f.write("%.3f" % percSpotCoverage[i])
            f.write("%\n")
        elif percSpotCoverage[i] == 0:
            f.write(line)
            f.write("%s \t\t\t| \n" % name)

    f.write(line)
    f.close()

    print("Done writing the CDI!")


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


def find_biggest_contour(imgs):
    '''
    Finds the biggest contour in an image.
    :return: The biggest contour
    '''