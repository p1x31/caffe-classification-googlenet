import pandas as pd
import numpy as np 
import imutils
import argparse
from cv2 import cv2
import os
from scipy.optimize import minimize
from matplotlib import pyplot as plt
from scipy.signal import convolve2d

#def V(theta, N):
#    return sum(a0*(cos(i*theta)) for i in range(1, N + 1))
# initial guesses

sum_sum_g_i_f_j = 0
sum_sum_f_i_f_j = 0

def every_pixel(currentFrame, previous):
    for y in range(1, previous.shape[1]-1):
        for x in range(1, previous.shape[0]-1):
            sum_sum_g_i_f_j =+ cv2.sumElems(currentFrame*previous[x,y])[0]
            sum_sum_f_i_f_j =+ cv2.sumElems(previous*previous[x,y])[0]
            #pix_sum =+ previous[x,y]
            #b = pix_sum - a*pix_sum
            #a = (pix_sum - b)/pix_sum
    return sum_sum_f_i_f_j, sum_sum_g_i_f_j

# Define the function to calculate the sum squared
def calcSumSquared(x, sum_pixels_f_i, sum_pixels_g_i, f_i_sq, f_i_g_i, N, g_i_sq):
    a = x[0]
    b = x[1]
    # All but the last value
    #f_i = x[:-1]
    f_i = sum_pixels_f_i
    g_i = sum_pixels_g_i
    f_i_sq = f_i_sq 
    f_i_g_i = f_i_g_i
    n = N
    g_i_sq = g_i_sq
    #sumSquared = sum((np.multiply(a, f_i)+b-f_i)**2)
    sumSquared = abs(a**2*f_i_sq+2*a*b*f_i-a*f_i_g_i+n*b**2-2*b*g_i+g_i_sq)
    return sumSquared

# Define objective function for optimization
def objective(x, sum_pixels_f_i, sum_pixels_g_i,f_i_sq, f_i_g_i, N, g_i_sq):
    return calcSumSquared(x, sum_pixels_f_i, sum_pixels_g_i, f_i_sq, f_i_g_i, N, g_i_sq)


# Define constraint for optimization (must equal to 0)
#a
def constraint1(x, sum_pixels_f_i, sum_pixels_g_i, f_i_sq, f_i_g_i, N, g_i_sq):
    a = x[0]
    b = x[1]
    # All but the last value
    #f_i = x[:-1]
    #f_i = x[2]
    f_i = sum_pixels_f_i
    f_i_sq = f_i_sq
    f_i_g_i = f_i_g_i
    #print(f_i)
    #return sum(a*f_i**2 + b*f_i - f_i**2)
    return (a*f_i_sq + b*f_i - f_i_g_i)
#b
def constraint2(x, sum_pixels_f_i, sum_pixels_g_i, f_i_sq, f_i_g_i, N, g_i_sq):
    a = x[0]
    b = x[1]
    # All but the last value
    #f_i = x[:-1]
    #f_i = x[2]
    f_i = sum_pixels_f_i
    g_i = sum_pixels_g_i
    n = N
    #return sum(a*f_i - f_i + b)
    return (a*f_i + n*b - g_i)

def toVector(previous):
    assert previous.shape == (393,700)
    return np.hstack([previous.flatten()])

def toMatrix(vec):
    assert vec.shape == (393*700,)
    return vec[:].reshape(393,700)
# params: source, output
def get_a_b_manual(previous, currentFrame):
    f_i_sq = cv2.sumElems(np.multiply(previous,previous))[0]
    #g_i_sq = cv2.sumElems(np.multiply(currentFrame,currentFrame))[0]
    f_i_g_i = cv2.sumElems(np.multiply(previous,currentFrame))[0]
    N = np.multiply(previous.shape[0],previous.shape[1])
    f_i = cv2.sumElems(previous)[0]
    g_i = cv2.sumElems(currentFrame)[0]
    a = (np.multiply(f_i,g_i) - np.multiply(N,f_i_g_i))/(np.multiply(f_i,f_i) - np.multiply(N,f_i_sq))
    b = (g_i-a*f_i)/N
    #print(f"pseudo a = {a}")
    #print(f"pseudo b = {b}")
    return a, b

def get_a_b(previous, currentFrame):
    a = 1
    b = 1
    #previous = toVector(previous)
    #f_i = np.mean(previous)
    f_i_sq = cv2.sumElems(np.multiply(previous,previous))[0]
    g_i_sq = cv2.sumElems(np.multiply(currentFrame,currentFrame))[0]
    f_i_g_i = cv2.sumElems(np.multiply(previous,currentFrame))[0]
    N = np.multiply(previous.shape[0],previous.shape[1])
    f_i = cv2.sumElems(previous)[0]
    g_i = cv2.sumElems(currentFrame)[0]
    #f_i = toVector(previous)
    #f_i = previous.flatten()
    #(sum_sum_g_i_f_j, sum_sum_f_i_f_j) = every_pixel(currentFrame, previous)
    #a_check = (N*f_i_g_i - sum_sum_g_i_f_j)/(N*f_i_sq - sum_sum_f_i_f_j)
    #b_check = (g_i-a_check*f_i)/N
    
    arguments = (f_i, g_i, f_i_sq, f_i_g_i, N, g_i_sq)

    # Load constraints into dictionary form
    con1 = ({"type": "eq", "fun":constraint1, 'args': arguments })
    con2 = ({"type": "eq", "fun":constraint2, 'args': arguments })
    #con3 = ({"type": "eq", "fun":constraint3})
    #cons = [con1,con2,con3]
    cons = [con1,con2]
    #p0 = np.array([f_i])
    x0 = np.asarray(([a,b]), dtype=object)
    #x0 = f_i
    sol = minimize(objective, x0, args=arguments,  method="SLSQP", constraints=cons, options={"disp":True})
    #vec = sol.x[2]
    #frame = toMatrix(vec)
    a = sol.x[0]
    b = sol.x[1]
    #mean = sol.x[2]
    #print(sol.fun)
    #print(f"a = {a}")
    #print(f"b = {b}")

    # a = 255 / (maximum_gray - minimum_gray)
    # b = -minimum_gray * a
    #a = np.divide((previous - b),previous)
    #b = np.multiply(previous,(1-a))
    return a, b

# Automatic brightness and contrast optimization with optional histogram clipping
# histogram equalization
def automatic_brightness_and_contrast(image, clip_hist_percent=1):
    gray = image

    # Calculate grayscale histogram
    # cv.calcHist(images, channels, mask, histSize, ranges[, hist[, accumulate]])
    # histSize For full scale, we pass [256].
    hist = cv2.calcHist([gray],[0],None,[256],[0,256])
    hist_size = len(hist)

    # Calculate cumulative distribution from the histogram
    accumulator = []
    accumulator.append(float(hist[0]))
    for index in range(1, hist_size):
        accumulator.append(accumulator[index -1] + float(hist[index]))

    # Locate points to clip
    maximum = accumulator[-1]
    clip_hist_percent *= (maximum/100.0)
    clip_hist_percent /= 2.0

    # Locate left cut
    minimum_gray = 0
    while accumulator[minimum_gray] < clip_hist_percent:
        minimum_gray += 1

    # Locate right cut
    maximum_gray = hist_size -1
    while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
        maximum_gray -= 1

    # Calculate alpha and beta values
    alpha = 255 / (maximum_gray - minimum_gray)
    beta = -minimum_gray * alpha

    '''
    # Calculate new histogram with desired range and show histogram 
    new_hist = cv2.calcHist([gray],[0],None,[256],[minimum_gray,maximum_gray])
    plt.plot(hist)
    plt.plot(new_hist)
    plt.xlim([0,256])
    plt.show()
    '''

    auto_result = cv2.convertTo(image, alpha=alpha, beta=beta)
    return (auto_result, alpha, beta)


if __name__ == '__main__':
    filename = "scene.avi"
    PATH = "./ForVadim/{}".format(filename)
    camera = cv2.VideoCapture(filename)
    ret, frame = camera.read()
    min_diff = None
    static_mask = None
    pix_mask_uint8 = None
    I = None
    I_2 = np.ones((3,3), dtype=np.uint8)
    I_3 = np.ones((3,3), dtype=np.uint8)
    size = (393,700)
    pix_mask = np.ones(size)
    # how many frames to average 1/alfa
    alfa = .50
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

            """
            # test put random matrix with known a and b and get those a b back
            ###
            if I is None:
                I = np.random.randint(255, size=size, dtype=np.uint8)
                continue
            print(I)
            #(a, b) = get_a_b_manual(I, I)
            #I = np.multiply(I,a)+b
            #print(I)
            a = 0.33
            b = 10
            I_1 =  np.multiply(I,a)+b
            print(I_1)
            (a, b) = get_a_b_manual(I_1, I)
            print("test:")
            print(a,b)
            I_1 = np.multiply(a,I_1)+b
            print(I_1)
            #one more multiply doubles the to 255
            I_1 = cv2.convertScaleAbs(I_1, alpha=a, beta=b)
            print(I_1)
            print(I.all() == I_1.all())
            diff = cv2.absdiff(I, I_1)
            cv2.imshow("diff", diff)
            """

            """
            # test 2 absdiff of the same frame
            (a, b) = get_a_b_manual(previous, previous)
            a = np.reciprocal(a)
            b = np.divide(b,a)
            enchanced_current = cv2.convertScaleAbs(previous, alpha=a, beta=b)
            _diff = cv2.absdiff(enchanced_current, enchanced_current)
            """
            
            """
            if I is None:
                I = np.random.randint(low = 99, high = 100, size=(3,3)).astype(dtype=np.uint8)
                continue
            I_2 = np.multiply(I_2,3)
            print(I)
            print(I_2)
            (a, b) = get_a_b_manual(I, I_2)
            #a = np.reciprocal(a)
            print(a)
            #b = -np.divide(b,a)
            print(b)
            I_3 = np.multiply(I,a)+b
            print(I_3)
            """
            
            # Contrast Limited Adaptive Histogram Equalization
            # deacrease clip limit to remove the noise
            # clipLimit	Threshold for contrast limiting.
            #clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(7,7))
            #cl1 = clahe.apply(grayClone)
            #cl2 = clahe.apply(previous)
            #auto_result, a, b = automatic_brightness_and_contrast(previous)
            #print(a,b)
            #cv2.imshow("Brightness", auto_result)
            #dst	= cv2.fastNlMeansDenoisingMulti(enchanced_current, enchanced_previous, 3, 7, 21)
            #dst_previous	= cv2.fastNlMeansDenoising(enchanced_previous, 1, 7, 21)
            # Calculate absolute difference of current frame and
            # the next frame
            _diff = cv2.absdiff(grayClone, previous)
            #_diff = cv2.absdiff(enchanced_current, dst_previous)


            #_diff = cv2.absdiff(cl1, cl2)
            #_diff = cv2.absdiff(grayClone, previous)
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

                # params: source, dest
                (a, b) = get_a_b_manual(grayClone, previous)
                #a = np.reciprocal(a)
                #print(a)
                #b = -np.divide(b,a)
                #print(b)
                # this makes grayClone -> previous
                enchanced_current = cv2.convertScaleAbs(grayClone, alpha=a, beta=b)
                #enchanced_previous = cv2.convertScaleAbs(previous, alpha=a, beta=b)

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
                #pixel_mean = np.mean(grayClone)
                # mean to change the threshold value
                #pixel_mean = np.mean(_diff)*100
                # if the difference in any pixel value more than 10 remove from background
                #pixel_mean = 10
                
                #true_mean = np.true_divide(_diff.sum(1),(_diff!=0).sum(1))[0]
                diff = cv2.absdiff(enchanced_current, previous)
                #pixel_mean = np.apply_along_axis(lambda v: np.median(v[np.nonzero(v)]), 0, diff)
                m = np.ma.masked_equal(diff, 0)
                pixel_mean = np.ma.median(m)*10
                #print(pixel_mean)
                
                # 0 difference -> No movement, True = 1 -> background
                pix_mask = (diff < pixel_mean).astype(np.int_)
                #convolve
                #kernel = [[0,1,0],[1,1,1],[0,1,0]]
                #pix_mask_filtered = convolve2d(pix_mask, kernel, mode='same')
                # less than 3 neighbours removed
                #pix_mask_filtered[pix_mask_filtered<=2] = 0
                #pix_mask_uint8 = np.multiply(pix_mask_filtered,255).astype(dtype=np.uint8)
                pix_mask_uint8 = np.multiply(pix_mask,255).astype(dtype=np.uint8)
                static_mask = np.multiply(np.float(alfa),static_mask) + np.multiply(np.float32(1.0 - alfa),np.multiply(previous,pix_mask))
            
            
            mask_uint8 = np.uint8(static_mask)
            # Treshold to binarize
            # (source image, treshold value, max value)
            # th, dframe = cv2.threshold(dframe, 30, 255, cv2.THRESH_BINARY)

            keypress_toshow = cv2.waitKey(1)
            
            if(keypress_toshow == ord("e")):
                show_pred = not show_pred
            
            cv2.imshow("Input frame",grayClone)

            #cv2.imshow("CLAHE", cl1)
            
            #cv2.imshow("enchanced", enchanced_current)

            #cv2.imshow("enchanced", dst_previous)
            
            cv2.imshow("pixel mask", pix_mask_uint8)
            
            cv2.imshow("Background", mask_uint8)
            
            #cv2.imshow("diff", _diff)

            keypress = cv2.waitKey(1) & 0xFF

            if(keypress == ord("q")):
                break
        else:
            print('Could not read frame')



camera.release()

cv2.destroyAllWindows()