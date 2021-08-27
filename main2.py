"""
TALLER 2 - PROCESAMIENTO IMAGEN Y VIDEO
PUJ
JUAN NICOLAS SOTO - ALISSON BERNAL
"""


import cv2
import numpy as np
from thetafilter import thetaFilter

if __name__ == '__main__':
    path_file = r'/Users/alissonsbernalcastro/Desktop/Master/Imagenes/01_1.tif'

T1 = thetaFilter(path_file)



# Implementación de banco de 4 filtros y reconstrucción de imagen con base a estos
theta = [0, 45, 90, 135]
c = 1
new_image = np.zeros_like(T1.image, dtype=float)
for the in theta:
    T1.set_theta(the, 30)
    name = f'img_{c}'
    name = T1.filtering()
    new_image += T1.filtering()
    cv2.imshow(f'Imagen_{the}', name)
    c += 1


new_image = new_image - new_image.min()
new_image = new_image / new_image.max()
print(new_image.max())
print(new_image.min())
cv2.imshow('Imagen Final', new_image)
cv2.imshow('Original' , T1.image)
cv2.waitKey(0)
