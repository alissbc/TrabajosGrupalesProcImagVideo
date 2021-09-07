"""
Taller 3 Alisson Bernal, Juan Nicolas Soto
Funciones para diezmado, Interpolado y filtrado de imagenes

"""

from Taller_3 import* # Se importan los métodos
import cv2
import numpy as np


# Cargado de la imagen
path_file = r'C:\Users\juann\Downloads/lena.png'
image = cv2.imread(path_file)

# Diezmado
img_dec = Decimate(image, 3)

# Interpolado
img_inter = Interpolate(image, 2)

# Filtrado (Banco de filtros)
filter, names = Filters(image)

# Proyección de imágenes
cv2.imshow('Original', image)
cv2.imshow('Decimate', img_dec)
cv2.imshow('Interpolate', img_inter)

## Imagenes del proceso de filtrado
i=0
for img in filter:
    cv2.imshow(f'Img_{names[i]}', img)
    i += 1

# 4. Interpolado de la imegen ILL
ILL_inter = Interpolate(filter[-1], 4)
cv2.imshow('ILL Interpolated', ILL_inter)

cv2.waitKey(0)