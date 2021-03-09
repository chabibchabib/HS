
######################################################################

afterImg = cv.imread('image2.png', 0).astype(float)
beforeImg = cv.imread('image1.png', 0).astype(float)

print('image0 shape',afterImg.shape)

u,v = Hs_pyram(beforeImg, afterImg,factor=2, alpha = 15, delta = 10**-1,Level=3,itmax=900,kernel_size=5,downsampling_gauss=0.80)
draw_quiver(u, v, beforeImg)



