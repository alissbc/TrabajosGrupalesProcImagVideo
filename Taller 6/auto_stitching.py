import cv2
import numpy as np
import os
import sys

'''
Taller 6 Alisson Bernal, Juan Nicolas Soto
Auto stitching
Este programa recibe una direccion de una carpeta via terminal con las imagenes ordenadas de izquierda a derecha y retorna una imagen panoramica centrada en una imagen
de referencia escogida por el usuario.

'''

# Contenedor de Homografias
H_t = []

# Ruta de carpeta
path = sys.argv[1]
#path = r'C:\Users\juann\PycharmProjects\pythonProject1\Proce_imag\Talleres\Taller_5\pictures'

# Cambiar Show a true si se desea ver la concordancia de puntos y las homografias
show = True
# show = sys.argv[2]

# Conteo de la cantidad de imagenes encontradas en la carpeta
N_p = len(os.listdir(path))
print(f'Numero de imagenes recibidas: {N_p}')



# Se le solicita al usuario que escoja la imagen de referencia, adicionalmente se valida de que la seleccion sea valida
while True:
    img_central = int(input(f'Seleccione cual de las {N_p} imagenes desea usar como imagen de referencia: '))
    if img_central <= N_p and img_central >= 1:
        break
    else:
        print(f'Numero de imagen de referencia debe estar entre 1 y {N_p}')


# Se le solicita al usuario que escoja el metodo de deteccion de keypoints entre ORB y SIFT
while True:
    metodo = int(input(f'Seleccione 1 para ORB y 2 para SIFT: '))
    if metodo == 1 or metodo == 2:
        break
    else:
        print(f'Seleccion no es valida')

# Ciclo for para calcular los keypoints matches y las homografias
for i in range(N_p - 1):

    # Cargue de las imagenes desde la carpeta indicada
    img_1_o = cv2.imread(os.path.join(path, os.listdir(path)[i]))
    img_2_o = cv2.imread(os.path.join(path, os.listdir(path)[i+1]))

    # Redimensinamiento de las imagenes para facilitar su observacion
    dim = (960, 540)
    img_1 = cv2.resize(img_1_o, dim, interpolation = cv2.INTER_AREA)
    img_2 = cv2.resize(img_2_o, dim, interpolation = cv2.INTER_AREA)

    # Se transforman imagenes a grises
    image_gray_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
    image_gray_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)

    # Se calculan los Keypoints
    if metodo == 1:
        orb = cv2.ORB_create()  # oriented FAST and Rotated BRIEF
        keypoints_1, descriptors_1 = orb.detectAndCompute(image_gray_1, None)
        keypoints_2, descriptors_2 = orb.detectAndCompute(image_gray_2, None)

        # Interest points matching
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(descriptors_1, descriptors_2)
    else:
        sift = cv2.SIFT_create()
        keypoints_1, descriptors_1 = sift.detectAndCompute(image_gray_1, None)
        keypoints_2, descriptors_2 = sift.detectAndCompute(image_gray_2, None)

        # Interest points matching
        bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
        matches = bf.match(descriptors_1, descriptors_2)



    # Se ordenan los matches segun su distancia, poniendo primero los mas robustos
    matches = sorted(matches, key=lambda x: x.distance)

    # Seleccion de maximo numero de matches
    number_matches = 100

    # Se dibujan matches sobre las imagenes
    image_matching = cv2.drawMatches(img_1, keypoints_1, img_2, keypoints_2, matches[:number_matches], None)
    if show:
        cv2.imshow('Imagen1', image_matching)
        cv2.waitKey(0)

    # Se cargan los keypoints en contenedores
    points_1 = []
    points_2 = []
    for i in range(number_matches):
        idx = matches[i].queryIdx
        idx2 = matches[i].trainIdx
        points_1.append(keypoints_1[idx].pt)
        points_2.append(keypoints_2[idx2].pt)


    pts1 = np.array(points_1)
    pts2 = np.array(points_2)

    # Calculo de homografias
    H, _ = cv2.findHomography(pts1, pts2, method=cv2.RANSAC)
    H_t.append(H)


# Contenedor de imagenes
st = []

#### Transformaciones --  Homografia ####
for i in range(N_p):
    # Homografias Izquierdas
    if i < img_central -1:
        j = i
        img = cv2.imread(os.path.join(path, os.listdir(path)[i]))
        while j < img_central - 1:
            img = cv2.warpPerspective(img, H_t[j], (img.shape[1], img.shape[0]))
            if show:
                cv2.imshow('Homograf Izq', img)
                cv2.waitKey(0)
            j += 1
    # Homografias derechas
    elif i > img_central - 1:
        j = i -1
        img = cv2.imread(os.path.join(path, os.listdir(path)[i]))
        while j >= img_central - 1:
            img = cv2.warpPerspective(img, np.linalg.inv(H_t[j]), (img.shape[1], img.shape[0]))
            if show:
                cv2.imshow('Homograf Der', img)
                cv2.waitKey(0)
            j -= 1
    # Se carga la imagen central al contenedor de imagenes transformadas st
    else:
        img = cv2.imread(os.path.join(path, os.listdir(path)[i]))

    st.append(img)

# Union de imagenes
avg_image = st[img_central - 1]

# filtrando puntos negros de la homografia para que coincidan con la imagen de referencia
for x in range(avg_image.shape[0]):
    for y in range(avg_image.shape[1]):
        for p in range(len(st)):
            if p == img_central - 1:
                pass
            if st[p][x, y, 0] == 0 and st[p][x, y, 1] == 0 and st[p][x, y, 2] == 0:
                st[p][x, y, 0] = avg_image[x, y, 0]
                st[p][x, y, 1] = avg_image[x, y, 1]
                st[p][x, y, 2] = avg_image[x, y, 2]

# Promediando imagenes
for i in range(len(st)):
    if i == img_central -1 :
        pass
    else:
        alpha = 1.0/(i + 1)
        beta = 1.0 - alpha
        avg_image = cv2.addWeighted(st[i], alpha, avg_image, beta, 0.0)

# Se graba imagen final
cv2.imwrite('Stitched_image.png', avg_image)

cv2.imshow('Imagen final', avg_image)
cv2.waitKey(0)
