import os
import argparse
import camera_cal as camcal
import precision_plot as pp
import rcnn_func as rcf
import functions_openCV as ftc


def parse_arguments():
    """
    Parses arguments for the program.

    :return: The arguments specified
    """
    parser = argparse.ArgumentParser(description='Image Classifier Predictions')

    # Command line arguments
    parser.add_argument('--google_colab', type=bool, default=False, help='Whether the program is running on Google Colab')
    parser.add_argument('--validation_folder', type=str, default="./fish_pics/rcnn_dataset/validation/", help='Folder containing the validation images after creating the dataset')
    parser.add_argument('--image_dir', type=str, default="./fish_pics/input_images/", help='Absolute path to images')
    parser.add_argument('--checkpoint', type=str,
                        default='./checkpoint.pth',
                        help='Path to checkpoint')
    parser.add_argument('--image_dir_rcnn_images', type=str, default="./fish_pics/rcnn_masks/images/", help='Absolute path to image folder')
    parser.add_argument('--image_dir_rcnn_annotations', type=str, default="./fish_pics/rcnn_masks/annotations/", help='Absolute path to annotation folder')
    parser.add_argument('--train_rcnn', type=bool, default=False, help='Train mask rcnn classifier')
    parser.add_argument('--run_prediction_model', type=bool, default=True, help='Classify undistorted images')
    parser.add_argument('--topk', type=int, default=5, help='Top k classes and probabilities')
    parser.add_argument('--json', type=str, default='classes_dictonary.json', help='class_to_name json file')
    parser.add_argument('--device', type=str, default='cuda', help='\'cuda\' for GPU or \'cpu\' for CPU')
    parser.add_argument('--arUco_marker_cur', type=float, default=19.2, help='ArUco marker circumference')
    parser.add_argument('--calibrate_cam', type=bool, default=False, help='Set to \'True\' to re-calibrate camera. '
                        'Remember to put images of checkerboard in calibration_imgs folder')
    parser.add_argument('--undistorted', type=bool, default=False, help='Classify undistorted images')
    parser.add_argument('--make_new_data_set', type=bool, default=False, help='Use images in fish_pics\input_images and create a new dataset')
    parser.add_argument('--model_name', type=str, default='model_1', help='Select the model that we want to use for instance segmentation')
    parser.add_argument('--num_epochs', type=int, default=5, help='number of epoch to run for training')

    arguments = parser.parse_args()

    return arguments


def main(args=None):

    # Load arguments
    arguments = parse_arguments()

    # Create dataset if requested
    if arguments.make_new_data_set:
        rcf.create_dataset_mask_rcnn(arguments)

    # Create path to model
    basedir = os.path.dirname(os.path.abspath(__file__))
    models_path = os.path.join(basedir, 'models')
    model_name = arguments.model_name
    model_path = os.path.join(models_path, model_name)
    
    # Check if we want to calibrate the camera
    if arguments.calibrate_cam:
        camcal.calibrate_camera(arguments)

    # Check if we want to run the RCNN trainer
    if arguments.train_rcnn:
        # Run the RCNN trainer
        rcf.run_rcnn_trainer(basedir, model_path, arguments.num_epochs)

    if arguments.run_prediction_model:

        print("Running prediction model...")
        img_names, img_normal, contours, precisions, labels = rcf.test_rcnn(basedir, model_path, use_morphology=False)

        # ArUco marker calibration for size estimation, displays results of the calculated size
        len_estimate, fish_names_sorted = ftc.load_ArUco_cali_objectsize_and_display(img_normal, img_names, contours, arguments, labels, precisions, display=True)

    # Precision calculation
    pp.calc_len_est_names(fish_names_sorted, len_estimate)


if __name__ == '__main__':
    main()
