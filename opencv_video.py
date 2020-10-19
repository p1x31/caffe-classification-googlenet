import pandas as pd
import numpy as np 
import imutils
import argparse
from cv2 import cv2
import os

def detect_image():
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", required=True,
       help="path to input video")
    args = vars(ap.parse_args())

    cap = cv2.VideoCapture('scene.avi')
    while(True):
        (_, image) = cap.read()
        image = imutils.resize(image, height = 400)
        #image = cv2.imread(args["image"])
        cv2.imshow("Orginal", image)

        keypress = cv2.waitKey(1) & 0xFF
        if(keypress == ord("q")):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_image()
    
