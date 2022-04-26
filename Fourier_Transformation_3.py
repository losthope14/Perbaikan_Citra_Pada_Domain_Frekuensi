import numpy as np
import cv2
import matplotlib.pyplot as plt

image_1 = cv2.imread('fuji.jpg',0)

def distance(point1, point2):
    return np.sqrt((point1[0]-point2[0])**2 + (point1[1] - point2[1])**2)

def gaussianLP(D0, imgShape):
    base = np.zeros(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows/2, cols/2)
    for x in range (cols):
        for y in range(rows):
            base[y,x] = np.exp(((-distance((y,x), center)**2)/(2*(D0**2))))
    return base

def gaussianHP(D0, imgShape):
    base = np.zeros(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows/2, cols/2)
    for x in range (cols):
         for y in range(rows):
            base[y,x] = 1- np.exp(((-distance((y,x), center)**2)/(2*(D0**2))))
    return base

original = np.fft.fft2(image_1)
center = np.fft.fftshift(original)

plt.figure(figsize=(6.54*5, 4.8*5), constrained_layout=False)
plt.subplot(131), plt.imshow(image_1, "gray"), plt.title("Original Image")

LowPassCenter = center * gaussianLP(50, image_1.shape)
LowPass = np.fft.ifftshift(LowPassCenter)
inverse_LowPass = np.fft.ifft2(LowPass)
plt.subplot(132), plt.imshow(np.abs(inverse_LowPass), "gray"), plt.title("Gaussian Low Pass")

HighPassCenter = center * gaussianHP(50, image_1.shape)
HighPass = np.fft.ifftshift(HighPassCenter)
inverse_HighPass = np.fft.ifft2(HighPass)
plt.subplot(133), plt.imshow(np.abs(inverse_HighPass), "gray"), plt.title("Gaussian High Pass")
plt.show()