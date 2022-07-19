import copy
import warnings

import cv2
import numpy as np


def undistort(inputImgs, k_1, k_2, imgCenterX, imgCenterY, Fx, Fy, show_img=False):
    '''
    Undistorts images using parameters found in MATLAB calibration using 'standard'.

    :param inputImgs: The distorted images
    :return: The undistorted image
    '''

    print("Started undistorting images...")

    undistortedImgs = []

    for img in inputImgs:
        h, w, ch = img.shape

        undistorted = np.zeros(img.shape, np.uint8)

        for y in np.arange(-1, 1, 1 / h):
            for x in np.arange(-1, 1, 1 / w):
                xorig = x
                yorig = y

                r = np.sqrt(xorig ** 2 + yorig ** 2)
                output_pos_x = round(Fx * (xorig * (1 + k_1 * r ** 2 + k_2 * r ** 4)) + imgCenterX);
                output_pos_y = round(Fy * (yorig * (1 + k_1 * r ** 2 + k_2 * r ** 4)) + imgCenterY);

                input_pos_x = round(Fx * x + imgCenterX)
                input_pos_y = round(Fy * y + imgCenterY)

                if input_pos_x < w - 1 and input_pos_y < h - 1 and output_pos_x < w - 1 and output_pos_y < h - 1:
                    if input_pos_x >= 0 and input_pos_y >= 0 and output_pos_x >= 0 and output_pos_y >= 0:
                        undistorted.itemset((input_pos_y, input_pos_x, 0), img.item((output_pos_y, output_pos_x, 0)))
                        undistorted.itemset((input_pos_y, input_pos_x, 1), img.item((output_pos_y, output_pos_x, 1)))
                        undistorted.itemset((input_pos_y, input_pos_x, 2), img.item((output_pos_y, output_pos_x, 2)))

        undistortedImgs.append(undistorted)

        if show_img:
            cv2.imshow("Undistorted img", undistorted)
            cv2.waitKey(1)

    print("Done with undistorting!")

    return undistortedImgs


def findInRange(hsv_img, range_hsv=[]):
    lowerH = (range_hsv[0][0], range_hsv[0][1])
    lowerS = (range_hsv[1][0], range_hsv[1][1])
    lowerV = (range_hsv[2][0], range_hsv[2][1])

    h, w, ch = hsv_img.shape[:3]

    segmentedImg = np.zeros((h, w), np.uint8)
    # We start segmenting
    for y in range(h):
        for x in range(w):
            H = hsv_img.item(y, x, 0)
            S = hsv_img.item(y, x, 1)
            V = hsv_img.item(y, x, 2)
            # If Hue lies in the lowerHueRange(Blue hue range) we want to segment it out
            if lowerH[0] <= H <= lowerH[1] and lowerS[0] <= S <= lowerS[1] and lowerV[0] <= V <= lowerV[1]:
                segmentedImg.itemset((y, x), 255)
            else:
                segmentedImg.itemset((y, x), 0)

    return segmentedImg


def crop(images, y, x, height, width):
    """
    Crops images to specified area

    :param images: An array of images to crop
    :param y: The Y value of the crop point
    :param x: The X value of the crop point
    :param height: The height of the cropped part
    :param width: The width of the cropped part
    :return: An array of cropped images
    """
    cropped_images = []
    for n in images:
        ROI = n[y:y + height, x:x + width]
        cropped_images.append(ROI)

    return cropped_images


def bitwise_and(img, mask):
    '''
    A bitwise operation to stitch a picture to a mask

    :param img: The image to reference in color
    :param mask: The mask to reference in grayscale
    :return: An image where the mask decides
    '''

    # Get the height and width of the image to make an array filled with zeroes to have a black image
    height, width = img.shape[:2]
    main_clone = np.zeros((height, width, 3), dtype=np.uint8)

    # Go through each pixel and change the clones pixel values to the ones of the original image, as long as the same
    # pixel on the mask is not black
    for y in range(height):
        for x in range(width):
            mask_val = mask.item(y, x)
            if mask_val != 0:
                main_clone.itemset((y, x, 0), img.item(y, x, 0))
                main_clone.itemset((y, x, 1), img.item(y, x, 1))
                main_clone.itemset((y, x, 2), img.item(y, x, 2))
            else:
                main_clone.itemset((y, x, 0), 0)
                main_clone.itemset((y, x, 1), 0)
                main_clone.itemset((y, x, 2), 0)

    return main_clone


def erosion(mask, kernel_ero):
    """
    A standard erosion solver, shrinks the given mask.

    :param mask: The mask to shrink
    :param kernel_ero: The kenerl to shrink the erosion by
    :return: Returns the erosied mask
    """

    # Acquire size of the image
    height, width = mask.shape[0], mask.shape[1]
    # Define the structuring element
    k = kernel_ero.shape[0]
    SE = np.ones((k, k), dtype=np.uint8)
    # kernel_ero = np.ones((k, k), dtype=np.uint8)
    constant = (k - 1) // 2

    # Define new image
    imgErode = np.zeros((height, width), dtype=np.uint8)

    # Erosion
    if k % 2 >= 1:
        for y in range(constant, height - constant):
            for x in range(constant, width - constant):
                temp = mask[y - constant:y + constant + 1, x - constant:x + constant + 1]
                product = temp * SE
                imgErode[y, x] = np.min(product)
    else:
        warnings.warn("Kernel shape is even, it should be uneven!")

    return imgErode


def dilation(mask, kernel_di):
    '''
    A standard dilation solver, expands the given mask.

    :param mask: The mask to dilate
    :param kernel_di: The kernel to dilate by
    :return: The dilated mask
    '''

    # Acquire size of the image
    height, width = mask.shape[0], mask.shape[1]
    # Define new image to store the pixels of dilated image
    imgDilate = np.zeros((height, width), dtype=np.uint8)
    # Define the kernel shape
    ks = kernel_di.shape[0]
    # Use that to define the constant for the middle part
    constant1 = (ks - 1) // 2
    # Dilation
    if ks % 2 >= 1:
        for y in range(constant1, height - constant1):
            for x in range(constant1, width - constant1):
                temp = mask[y - constant1:y + constant1 + 1, x - constant1:x + constant1 + 1]
                product = temp * kernel_di
                imgDilate[y, x] = np.max(product)
    else:
        warnings.warn("Kernel shape is even, it should be uneven!")

    return imgDilate


def morph_close(mask, kernel):
    """
    Close morphology on a mask and a given kernel.

    :param kernel:
    :param mask: The mask to use the morphology on
    :return: Close morphology on a mask
    """

    dilate = dilation(mask, kernel)
    ero = erosion(dilate, kernel)

    return ero


def morph_open(mask, kernel):
    """
    Open morphology on a mask and a given kernel.

    :param mask: The mask to use the morphology on
    :return: Open morphology on a mask
    """
    ero = erosion(mask, kernel)
    dilate = dilation(ero, kernel)

    return dilate


def grayScaling(img):
    """
    Function that will convert a BGR image to a mean valued greyscale image.
    :param img: BGR image that will be converted to greyscale
    :return: The converted greyscale image.
    """

    # Get the height and width of the image to create a cop of the other image in an array of zeros
    h, w, = img.shape[:2]
    greyscale_img1 = np.zeros((h, w, 1), np.uint8)

    # Go through each pixel in the image and record the intensity, then safe it for the same pixel in the image copy
    for y in range(h):
        for x in range(w):
            I1 = (img.item(y, x, 0) + img.item(y, x, 1) + img.item(y, x, 2)) / 3
            greyscale_img1.itemset((y, x, 0), I1)
    return greyscale_img1


def convert_RGB_to_HSV(img):
    """
    Converts an RGB image to HSV.

    :param img: The image to convert
    :return: HSV image
    """

    width, height, channel = img.shape

    # Get each color channel of the image
    B, G, R = img[:, :, 0] / 255, img[:, :, 1] / 255, img[:, :, 2] / 255

    # Create a clone of the other image with the same height and width
    hsv_img = np.zeros(img.shape, dtype=np.uint8)

    # For each pixel in the image
    for i in range(width):
        for j in range(height):

            # Defining Hue
            h, s, v = 0.0, 0.0, 0.0
            r, g, b = R[i][j], G[i][j], B[i][j]

            max_rgb, min_rgb = max(r, g, b), min(r, g, b)
            dif_rgb = (max_rgb - min_rgb)

            if r == g == b:
                h = 0
            elif max_rgb == r:
                h = ((60 * (g - b)) / dif_rgb)
            elif max_rgb == g:
                h = (((60 * (b - r)) / dif_rgb) + 120)
            elif max_rgb == b:
                h = (((60 * (r - g)) / dif_rgb) + 240)
            if h < 0:
                h = h + 360

            # Defining Saturation
            if max_rgb == 0:
                s = 0
            else:
                s = ((max_rgb - min_rgb) / max_rgb)

            # Defining Value
            hsv_img[i][j][0], hsv_img[i][j][1], hsv_img[i][j][2] = h / 2, s * 255, s * 255

    return hsv_img


def segment_cod(images, show_images=False):
    """
    Segments the cod from the background in each image using HSV.

    :param images: The images to isolate the cods from
    :param show_images: Tells the function whether or not display images. Mostly used for debugging
    :return:
    """

    print("Started segmenting the cods...")

    inRangeImages = []
    segmentedImages = []

    for img in images:

        # Convert image to HSV from BGR
        hsv_img = copy.copy(img)

        hsv_img = convert_RGB_to_HSV(hsv_img)

        # Define the lower hue value to be blue.
        lowerH = (99, 117)

        h, w, ch = hsv_img.shape[:3]

        mask = np.zeros((h, w), np.uint8)
        # We start segmenting
        for y in range(h):
            for x in range(w):
                H = hsv_img.item(y, x, 0)
                # If Hue lies in th lowerHueRange(Blue hue range) we want to segment it out
                if lowerH[1] > H > lowerH[0]:
                    mask.itemset((y, x), 0)
                else:
                    mask.itemset((y, x), 255)

        # For easier representation
        segmented_cod = bitwise_and(img, mask)

        if show_images:
            cv2.imshow("res", segmented_cod)
            cv2.imshow("mask", mask)

            key = cv2.waitKey(1)
            if key == 27:
                break

        # add to array
        inRangeImages.append(mask)
        segmentedImages.append(segmented_cod)

    print("Finished segmenting the cods!")

    return inRangeImages, segmentedImages
