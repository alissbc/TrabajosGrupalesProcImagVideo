"""
TALLER 2 - PROCESAMIENTO IMAGEN Y VIDEO
PUJ
JUAN NICOLAS SOTO - ALISSON BERNAL
"""

import cv2
import numpy as np

class thetaFilter:

    # Constructor --- Recibe la ruta de la imagen y la almacena en el objeto image
    def __init__(self, path_i):
        self.path = path_i
        print(self.path)
        self.image = cv2.imread(self.path, cv2.IMREAD_GRAYSCALE)
        self.theta = 0
        self.delta_theta = 0

    # Método para recibir los parámetros theta y delta_theta
    def set_theta(self, theta, delta_theta):
        self.theta = theta
        self.delta_theta = delta_theta

    # Método de filtrado via FFT
    def filtering(self):
        image_gray_fft = np.fft.fft2(self.image)
        image_gray_fft_shift = np.fft.fftshift(image_gray_fft)

        # fft visualization
        image_gray_fft_mag = np.absolute(image_gray_fft_shift)
        image_fft_view = np.log(image_gray_fft_mag + 1)
        image_fft_view = image_fft_view / np.max(image_fft_view)
        filter_mask = np.zeros_like(image_fft_view)
        half_x = int(filter_mask.shape[0] / 2)
        half_y = int(filter_mask.shape[1] / 2)

        # Dado que 0º, 180º y -180º indican una orientación vertical se decide trabajar como una equivalencia de 0
        if self.theta % 180 == 0:
            self.theta = 0

        # Definición inicial de puntos a theta + delta_theta y theta - delta_theta
        Px1 = -half_x
        Px2 = half_x
        Py1 = np.tan((self.theta + self.delta_theta + 0.1) * np.pi / 180)*Px1
        Py2 = np.tan((self.theta + self.delta_theta + 0.1) * np.pi / 180)*Px2
        Py3 = np.tan((self.theta - self.delta_theta + 0.1) * np.pi / 180) * Px1
        Py4 = np.tan((self.theta - self.delta_theta + 0.1) * np.pi / 180) * Px2

        # Generación de máscara de filtrado para caso 0º
        if self.theta == 0:
            for i in range(0,filter_mask.shape[0]):
                for j in range(int(Py1+half_x), int(Py3 + half_x)):
                    filter_mask[i][j] = 1

        # Generación de máscara de filtrado para casos diferentes a vertical y horizontal
        if self.theta != 0 and self.theta != 90:
            for j in range(0, filter_mask.shape[1]):
                y = int((Py3 + (j-half_y - Px1) * (Py2-Py3)/(Px2-Px1)) + half_y)
                y1 = int((Py1 + (j-half_y - Px1) * (Py1 - Py4) / (Px1 - Px2)) + half_y)
                #print(j, y, y1)
                for i in range(0, filter_mask.shape[0]):
                    if i <= y and i >= y1:
                        filter_mask[i][j] = 1

        # Generación de máscara de filtrado para caso 90º
        if self.theta == 90:
            Py1 = np.tan((0 + self.delta_theta + 0.1) * np.pi / 180) * Px1
            Py3 = np.tan((0 - self.delta_theta + 0.1) * np.pi / 180) * Px1
            for i in range(0,filter_mask.shape[0]):
                for j in range(int(Py1+half_x), int(Py3 + half_x)):
                    filter_mask[j][i] = 1

        # Aplicación del filtro con la máscara sobre la imagen fft
        filter_image_fft = filter_mask * image_fft_view
        fft_filtered = image_gray_fft_shift * filter_mask
        image_filtered = np.fft.ifft2(np.fft.fftshift(fft_filtered))
        image_filtered = np.absolute(image_filtered)
        image_filtered /= np.max(image_filtered)
        # cv2.imshow('Imagen_fft', image_fft_view)
        # cv2.imshow('Imagen_filtrada', filter_image_fft)
        # cv2.imshow("Image", image_filtered)
        # cv2.waitKey(0)
        return image_filtered


