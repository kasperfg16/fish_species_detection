import cv2
import os
import rcnn_func as rcf
import numpy as np


def save_coefficients(mtx, dist):

    """ Save the camera matrix and the distortion coefficients to given path/file. """

    # Getting the directory of the cali_matlab file
    basedir = os.path.dirname(os.path.abspath(__file__))
    cali_file = '/cali_matlab.yaml'
    path = basedir + cali_file

    print("Saving camera matrix and distortion coefficients to {}".format(path))

    # Open and write to file
    cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
    cv_file.write("K", mtx)
    cv_file.write("D", dist)
    cv_file.release()


def load_coefficients():

    '''Loads the coefficients from a calibration file.'''

    # Getting the directory of the cali_matlab file
    basedir = os.path.dirname(os.path.abspath(__file__))
    cali_file = '/cali_matlab.yaml'
    path = basedir + cali_file

    # FILE_STORAGE_READ
    cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    camera_matrix = cv_file.getNode("K").mat()
    dist_matrix = cv_file.getNode("D").mat()
    cv_file.release()

    return camera_matrix, dist_matrix


def calibrate_camera(displayCorners=False):

    """
    Calibrates a camera using checkerboard pattern (6x9).

    :return: Saves camera parameters in an YAML file
    """

    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((6 * 9, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

    # Load in the images for calibration
    imgs = rcf.load_images_from_folder("calibration_imgs")

    print("Found {} images for image calibration".format(len(imgs)))

    # Display image to see if the first image was read correctly and is grayscale
    cv2.imshow("Distorted examples image", imgs[0])
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