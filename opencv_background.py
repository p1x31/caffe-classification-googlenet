import pandas as pd
import numpy as np 
import imutils
import argparse
from cv2 import cv2
import os

if __name__ == '__main__':
    filename = "scene.avi"
    PATH = "./ForVadim/{}".format(filename)
    camera = cv2.VideoCapture(filename)
    ret, frame = camera.read()
    min_diff = None
    static_mask = None
    size = (393,700)
    pix_mask = np.ones(size)
    a = .99
    # loop until interrupted
    while (True):
        if ret:
            assert not isinstance(frame,type(None))
            previous = frame[:]
            gray = cv2.cvtColor(previous,cv2.COLOR_BGR2GRAY)
            previous = imutils.resize(gray, width = 700)
            if static_mask is None:
                static_mask = previous
                continue
            
            (grabbed,frame) = camera.read()
            frame = imutils.resize(frame, width = 700)
            clone = frame.copy()
            
            (height,width) = frame.shape[:2]
            
            #potential speedup if converted to grayscale, the net should intake grayscale as well
            grayClone = cv2.cvtColor(clone,cv2.COLOR_BGR2GRAY)
            
            # Calculate absolute difference of current frame and
            # the next frame
            
            _diff = cv2.absdiff(grayClone, previous)
            num_diff = cv2.countNonZero(_diff)
            
            # Calculate the minimum nonzero pixels in order to update the background
            if min_diff is None:
                min_diff = np.min(num_diff)
                continue
            min_diff = np.min(num_diff) if min_diff > np.min(num_diff) else min_diff
        
            if num_diff > min_diff:
                # gives coordinates in (x,y) format. Note that, row = x and column = y.
                # mask = cv2.findNonZero(_diff)
                # rows = 79844; cols = 1
                # pix_mask[tuple(mask.T)] = 0


                #for i in range (0, num_rows):
                #    for j in range(0, num_cols):
                #        if(mask(i,j) != None):
                #            pix_mask(0, :)=[0,j,i]
                # for i in range(0, num_rows):
                #     for j in range(0, num_cols):
                #        # if tuple(i,j) =! 0:
                #             pix_mask[i][j] = 1
                #        else: 
                #             pix_mask[i][j] = 0

                #pixel_mean = np.sum(static_mask,axis=(0,2,3))/(static_mask.shape[2]*static_mask.shape[3])
                #diff_mean = np.sum(_diff, axis=(0,1))/(_diff.shape[0]*_diff.shape[1])
                pixel_mean = np.mean(grayClone)
                print(pixel_mean)
                # 0 difference -> No movement, True = 1 -> background
                pix_mask = (_diff < pixel_mean).astype(np.int_)

                static_mask = np.multiply(np.float(a),static_mask) + np.multiply(np.float32(1.0 - a),np.multiply(previous,pix_mask))
            
            
            mask_uint8 = np.uint8(static_mask)
            # Treshold to binarize
            # (source image, treshold value, max value)
            # th, dframe = cv2.threshold(dframe, 30, 255, cv2.THRESH_BINARY)
            # label, prob = predict_frame(frame)

            keypress_toshow = cv2.waitKey(1)
            
            if(keypress_toshow == ord("e")):
                show_pred = not show_pred
            
            cv2.imshow("GrayClone",grayClone)
            
            cv2.imshow("Background", mask_uint8)

            # cv2.imshow("Video Feed", fgmask)

            keypress = cv2.waitKey(1) & 0xFF

            if(keypress == ord("q")):
                break
        else:
            print('Could not read frame')
    
camera.release()

cv2.destroyAllWindows()