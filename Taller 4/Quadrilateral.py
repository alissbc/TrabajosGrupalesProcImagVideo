import numpy as np
import random
import cv2

class Quadrilateral:

    ##### Constructor --- Recibe y almacena el tamano de la imagen
    def __init__(self, n):
        self.N = n
        # Se revisa la condicion de que N sea par. Si no retorna mensaje de error
        if self.N % 2 == 0:
            pass
        else:
            print('N debe ser numero par. Intente de nuevo')

    '''
    Funcion para crear una imagen de NxN de color cian con un cuadrilatero de color magenta cuyos vertices se encuentran en cada uno de los cuadrantes de la imagen y son generados 
    aleatoriamente con una distribucion uniforme 
    '''
    def generate(self):
        if self.N % 2 == 0: # Se valida condicion de numero par.
            # Se crea la imagen de NxN
            image = np.zeros((self.N, self.N, 3), np.uint8)
            image[0:self.N, 0:self.N] = (255, 255, 0)  # Se asigna color CIAN a la imagen

            # Definicion de vertices del cuadrilatero usando distribucion uniforme
            quad_point_1 = (int(random.uniform(0, self.N / 2)), int(random.uniform(0, self.N / 2)))
            quad_point_2 = int(random.uniform(self.N / 2, self.N)), int(random.uniform(self.N / 2, self.N))

            #Se asigna color magenta al cuadrilatero
            color = (135, 0, 245)

            # Dibujado de cuadrilatero
            image = cv2.rectangle(image, quad_point_1, quad_point_2, color, cv2.FILLED)

            # cv2.imshow('Imagen', image)
            # cv2.waitKey(0)

            return image


