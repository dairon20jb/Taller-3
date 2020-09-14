import numpy as np
import cv2
from time import time
class Filtrado: # Se crea la clase imageShape
    def __init__(self,image_gray_noisy): #Constructor definición
        self.image_gray_noisy = image_gray_noisy
        self.N = 7     # window size
    def gaussian_lp(self):
        image_gauss_lp = cv2.GaussianBlur(self.image_gray_noisy, (self.N, self.N), 1.5, 1.5)
        return image_gauss_lp
    def gaussian_hp(self):
        kernel_gauss = cv2.getGaussianKernel(self.N, 2.5)
        kernel_gauss_2D = np.multiply(kernel_gauss.T, kernel_gauss)
        impulse_2D = np.zeros((self.N, self.N), dtype=kernel_gauss.dtype)
        NN = int((self.N - 1) / 2)
        impulse_2D[NN, NN] = 1
        kernel_gauss_2D_hp = impulse_2D - kernel_gauss_2D
        image_gauss_hp = cv2.filter2D(self.image_gray_noisy, -1, kernel_gauss_2D_hp)
        return image_gauss_hp
    def median(self):
        image_median = cv2.medianBlur(self.image_gray_noisy, self.N)
        return image_median
    def bilateral(self):
        image_bilateral = cv2.bilateralFilter(self.image_gray_noisy, 15, 25, 25)
        return image_bilateral
    def nlm(self):
        image_nlm = cv2.fastNlMeansDenoising(self.image_gray_noisy, 5, 15, 25)
        return image_nlm
    def image_noise(self,image_noisy,image_filtered):
        image_noise = abs(image_noisy-image_filtered)
        return image_noise
    def tiempo(self,funcion):  # Medición de ejecución para x función
        start = time() #Valor inicial timer
        funcion_R= funcion()
        end = time() #Valor final del timer
        return ((end - start),funcion_R)
    def R_ECM(self,im_gray,im_filtrada): # Raíz del ECM
        RECM=  np.sqrt((np.square(im_gray - im_filtrada)).mean())
        return(RECM)