import math
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import os


def calc_len_est_names(image_names, len_estimate_list):
    
    # Avoid backend errors while displaying
    matplotlib.use('TkAgg')

    print('Comparing lenght estimate with ground truth')
    img_num_list = []
    len_real_list = []

    # Load seacreatures.xlsx
    basedir = os.path.dirname(os.path.abspath(__file__))
    aruco_marker_img_name = '/seacreatures.xlsx'
    aruco_marker_img_path = basedir + aruco_marker_img_name
    df = pd.read_excel(aruco_marker_img_path)

    # Make list of photo numbers e.g: GOPRO0012.jpg -> 12
    for img_path in image_names:
        
        img_name_jpg = os.path.split(img_path)[-1]
        img_number_tuple = img_name_jpg.split('.')

        img_name= img_number_tuple[0]
        img_number = int(img_name[-4:])

        print("Image number: {}".format(img_number))

        img_num_list.append(img_number)
    
    # If the image number matches a number in the "Photo" seacreatures.xlsx take the corresponding "Lenght"
    # In seacreatures.xlsx there is only every second number e.g 12, 14 ..., but we have image numbers 12,13,14 ...
    for img_number in img_num_list:
        for i in range(len(df['Photo'])):

            number_photo = df['Photo'].iloc[i]
            real_length = df['Lenght'].iloc[i]

            if  number_photo == img_number:
                len_real_list.append(real_length)
                continue
            elif img_number == number_photo + 1:
                len_real_list.append(real_length)
                continue

    # Have used this website for formulas: https://www.wikihow.com/Calculate-Precision
    len_real_array = np.array(len_real_list)
    len_estimate_array = np.array(len_estimate_list)
    
    # Absolute deviation
    abs_dev = abs(len_real_array - len_estimate_array)

    len_of_array = len(abs_dev)

    remove_list = []
    for i in range(0, len_of_array-1):
        if abs_dev[i] > 15:
            remove_list.append(i)
            print(remove_list)

    abs_dev = np.delete(abs_dev, remove_list)
    len_estimate_array = np.delete(len_estimate_array, remove_list)
    len_real_array = np.delete(len_real_array, remove_list)

    # Average deviation
    avg_dev = np.mean(abs_dev)
    error_array = len_real_array - len_estimate_array
    # Standard deviation
    std_dev = math.sqrt(sum(pow(error_array,2))/(len(error_array)-1))
    # Median deviation
    median = np.median(abs_dev)
    # Max error
    max_error = max(abs_dev)

    ##############################################
    #  Experiment

    # Deviation avg
    dev = len_real_array - len_estimate_array

    # Average deviation
    dev_avg = np.mean(dev)

    len_estimate_array = len_estimate_array + dev_avg

    file1 = open("dev_avg.txt", "w") 
    file1.write("dev_avg = " + str(dev_avg))
    file1.close()

    # Absolute deviation
    abs_dev = abs(len_real_array - len_estimate_array)
    # Average deviation
    avg_dev = np.mean(abs_dev)
    error_array = len_real_array - len_estimate_array
    # Standard deviation
    std_dev = math.sqrt(sum(pow(error_array,2))/(len(error_array)-1))
    # Median deviation
    median = np.median(abs_dev)
    # Max error
    max_error = max(abs_dev)

    ##############################################

    # Create plots
    plt.close('all')

    ## Fig 1
    fig1, ax1 = plt.subplots(2)
    fig1.set_dpi(100)
    ax1[0].set_title('Real vs Estimated length')
    ax1[0].set_xlabel('Real length [cm]')
    ax1[0].set_ylabel('Estimated length [cm]')
    ax1[0].scatter(len_real_array, len_estimate_array, c="blue")
    x = [min(len_real_array)-(len_real_array/100)*10, max(len_real_array)+(len_real_array/100)*10]
    ax1[0].plot(x, x, linestyle='-', c='green', label='Perfect estimation')

    handles, labels = ax1[0].get_legend_handles_labels()
    handles = handles[len(handles)-1:]
    labels = labels[len(labels)-1:]
    ax1[0].legend(handles=handles, labels=labels, loc='upper left', shadow=True, fontsize='large')

    textstr = 'Sample size: ' + str(len(abs_dev))
    textstr = textstr + '\n' + 'Average deviation: \u00B1' + str(avg_dev) + ' cm'
    textstr = textstr + '\n' + 'Standard deviation: \u00B1' + str(std_dev) + ' cm'
    textstr = textstr + '\n' + 'Median deviation: \u00B1' + str(median) + ' cm'
    textstr = textstr + '\n' + 'Max error: \u00B1' + str(max_error) + ' cm'

    ax1[1].text(0, 0, textstr, fontsize=14)
    ax1[1].axis('off')

    ## Fig 2
    fig2, ax2 = plt.subplots()
    fig2.set_dpi(100)
    ax2.set_title('Error at different lengths')
    ax2.set_xlabel('Real length [cm]')
    ax2.set_ylabel('Error [cm]')
    ax2.scatter(len_real_array, error_array, c="blue")
    x = [min(len_real_array)-(len_real_array/100)*10, max(len_real_array)+(len_real_array/100)*10]
    y = [0,0]
    ax2.plot(x, y, linestyle='-', c='green', label='Zero error')

    handles, labels = ax2.get_legend_handles_labels()
    handles = handles[len(handles)-1:]
    labels = labels[len(labels)-1:]
    ax2.legend(handles=handles, labels=labels,loc='upper left', shadow=True, fontsize='large')

    plt.show()