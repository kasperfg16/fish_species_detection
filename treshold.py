import functions_openCV as ftc


# Threshold to create a mask for each image
mask_cod, segmented_images = ftc.segment_codOPENCV(cropped_images)
mask_cod_CLAHE, img_segmented_cod_CLAHE = ftc.segment_cod_CLAHEOPENCV(CLAHE)