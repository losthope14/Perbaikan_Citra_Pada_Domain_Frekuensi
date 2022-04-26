import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.color import rgb2gray

# simple averaging filter without scaling parameter
mean_filter = np.ones((3,3))

# creating a guassian filter
x = cv2.getGaussianKernel(5,10)
gaussian = x*x.T

# different edge detecting filters
# scharr in x-direction
scharr = np.array([[-3, 0, 3],
                   [-10,0,10],
                   [-3, 0, 3]])
# sobel in x direction
sobel_x= np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]])
# sobel in y direction
sobel_y= np.array([[-1,-2,-1],
                   [0, 0, 0],
                   [1, 2, 1]])
# laplacian
laplacian=np.array([[0, 1, 0],
                    [1,-4, 1],
                    [0, 1, 0]])

filters = [mean_filter, gaussian, laplacian, sobel_x, sobel_y, scharr]
filter_name = ['mean_filter', 'gaussian','laplacian', 'sobel_x', \
                'sobel_y', 'scharr_x']
fft_shift = [np.fft.fftshift(np.fft.fft2(x)) for x in filters]
mag_spectrum = [np.log(np.abs(z)+1) for z in fft_shift]

for i in range(6):
    plt.subplot(2,3,i+1),plt.imshow(mag_spectrum[i],cmap = 'gray')
    plt.title(filter_name[i]), plt.xticks([]), plt.yticks([])

img = cv2.imread('fuji.jpg')
gray = rgb2gray(img)
gray_x = np.fft.ifftshift(gray)

sobelx1 = cv2.Sobel(gray_x,cv2.CV_64F,1,0,ksize=5) # x-axis
sobely1 = cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=5) # y-axis

sobelX = np.fft.ifftshift(sobelx1)

plt.figure()
plt.imshow(np.log(abs(sobelX)+1), 'gray')
plt.title('Sobel Horizontal')

plt.figure()
plt.imshow(np.log(abs(sobely1)+1), 'gray')
plt.title('Sobel Vertical')
plt.show()