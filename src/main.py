import numpy as np
import cv2
import tensorflow as tf
from tf_bodypix.api import download_model, load_model, BodyPixModelPaths

bodypix_model = load_model(download_model(
    BodyPixModelPaths.MOBILENET_FLOAT_50_STRIDE_16
))

cap = cv2.VideoCapture(3)

while(True):
    ret, frame = cap.read()
   

    #image_array = tf.keras.preprocessing.image.img_to_array(frame)
    result = bodypix_model.predict_single(frame)
    mask = result.get_mask(threshold=0.75)
    mask2 = tf.keras.preprocessing.image.img_to_array(mask)
    mask2 = mask2.astype(np.uint8)


    res = cv2.bitwise_and(frame,frame,mask = mask2)


    cv2.imshow('mask',res)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()