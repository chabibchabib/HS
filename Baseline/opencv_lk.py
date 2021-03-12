import numpy as np
import cv2 as cv
import argparse

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
# Create some random colors
color = np.random.randint(0,255,(100,3))
# Take first frame and find corners in it

# Create a mask image for drawing purposes
beforeImg = cv.imread('img_000000_18.02244.tiff', 0).astype(float)
afterImg = cv.imread('img_000050_30.42245.tiff', 0).astype(float)

    # calculate optical flow
p1, st, err = cv.calcOpticalFlowPyrLK(beforeImg, afterImg,**lk_params)
   