import pandas as pd
import os

def calc_len_est(img_list_abs_path):
    df = pd.read_excel('./seacreatures.xlsx')

    for img in img_list_abs_path:

        img
        
        print (os.path.split(img)[-1])
        
        # df['Lenght'].iloc[1]


        # number_photo = df['Photo'].iloc[1]
        # lenght = df['Lenght'].iloc[1]

        # if  number_photo >= 


