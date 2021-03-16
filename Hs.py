import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage.filters import convolve as filter2
import os
#compute derivatives of the image intensity values along the x, y, time
def image_derivatives(img1, img2):
    # Computing Image derivatives 
    Gx = np.array([[-1, 1], [-1, 1]]) * 0.25
    Gy = np.array([[-1, -1], [1, 1]]) * 0.25
    Gt = np.ones((2, 2)) * 0.25
    fx = filter2(img1,Gx) + filter2(img2,Gx)
    fy = filter2(img1, Gy) + filter2(img2, Gy)
    ft = filter2(img1, -Gt) + filter2(img2, Gt)
    return [fx,fy, ft]

def get_magnitude(u, v):
    scale = 3
    sum = 0.0
    counter = 0.0

    for i in range(0, u.shape[0], 8):
        for j in range(0, u.shape[1],8):
            counter += 1
            dy = v[i,j] * scale
            dx = u[i,j] * scale
            magnitude = (dx**2 + dy**2)**0.5
            sum += magnitude

    mag_avg = sum / counter

    return mag_avg

def draw_quiver(u,v,beforeImg):
    scale = 3
    ax = plt.figure().gca()
    ax.imshow(beforeImg, cmap = 'gray')

    magnitudeAvg = get_magnitude(u, v)

    for i in range(0, u.shape[0], 8):
        for j in range(0, u.shape[1],8):
            dy = v[i,j] * scale
            dx = u[i,j] * scale
            magnitude = (dx**2 + dy**2)**0.5
            #draw only significant changes
            if magnitude > magnitudeAvg:
                ax.arrow(j,i, dx, dy, color = 'red')

    plt.draw()
    plt.show()
#input: images name, smoothing parameter, tolerance
#output: images variations (flow vectors u, v)
#calculates u,v vectors and draw quiver
def computeHS(beforeImg, afterImg, alpha, delta,itmax):
    #removing noise
    beforeImg  = cv2.GaussianBlur(beforeImg, (5, 5), 0)
    afterImg = cv2.GaussianBlur(afterImg, (5, 5), 0)

    # set up initial values
    u = np.zeros((beforeImg.shape[0], beforeImg.shape[1]))
    v = np.zeros((beforeImg.shape[0], beforeImg.shape[1]))
    fx, fy, ft = image_derivatives(beforeImg, afterImg)
    avg_kernel = np.array([[1 / 12, 1 / 6, 1 / 12],
                            [1 / 6, 0, 1 / 6],
                            [1 / 12, 1 / 6, 1 / 12]], float)
    iter_counter = 0
    while True:
        iter_counter += 1
        u_avg = filter2(u, avg_kernel)
        v_avg = filter2(v, avg_kernel)
        p = fx * u_avg + fy * v_avg + ft
        d = 4 * alpha**2 + fx**2 + fy**2
        prev = u

        u = u_avg - fx * (p / d)
        v = v_avg - fy * (p / d)

        diff = np.linalg.norm(u - prev, 2)
        print('erreur=',diff)
        if  diff < delta or iter_counter > itmax:
            print("iteration number: ", iter_counter)
            break


    return [u, v]

beforeImg = cv2.imread('img_000000_18.02244.tif', 0).astype(float)
afterImg = cv2.imread('img_000050_30.42245.tif', 0).astype(float)


u,v = computeHS(beforeImg, afterImg, alpha = 15, delta = 2*10**-1,itmax=800)

draw_quiver(u, v, beforeImg)
