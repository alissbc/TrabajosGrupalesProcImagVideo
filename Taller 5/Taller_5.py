import cv2
import numpy as np
import os
import sys

""" 
Taller 5 Alisson Bernal, Juan Nicolas Soto
Funciones para diezmado, Interpolado y filtrado de imagenes
"""

#se crean vectores vacios para almacenar los puntos de referencia en las imagenes y las homografias
points_g = []
H_t = []

def click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))

#definicion del path donde se encuentran las imagenes a analizar por parte del usuario, las imagenes deben estar organizadas
path = sys.argv[1]
#conteo de las imagenes disponibles en la direccion indicada
N_p = len(os.listdir(path))

#se informa el numero de imagenes recibidas
print(f'Numero de imagenes recibidas: {N_p}')

while True:
    #se solicita al usuario cual de las N imagenes desea usar como imagen de referencia
    img_central = int(input(f'Seleccione cual de las {N_p} imagenes desea usar como imagen de referencia: '))
    #se verifica que el numero de la imagene de referencia indicada sea mayor o igual que 1 y menor o igual que el numero maximo de imagenes disponibles
    if img_central <= N_p and img_central >= 1:
        break
    else:
        #se informa si el numero de la imagen de referencia no hace sentido
        print(f'Numero de imagen de referencia debe estar entre 1 y {N_p}')

#se cargan y visualizan las imagenes de dos en dos, 1 y 2, 2 y 3, ... se reciben los puntos de referencia entre imagenes indicados por el usuario
for i in range(N_p - 1):
    cv2.namedWindow("Image")
    cv2.setMouseCallback("Image", click)

    #se carga cada imagen y su respectiva siguiente imagen
    img_1_o = cv2.imread(os.path.join(path, os.listdir(path)[i]))
    img_2_o = cv2.imread(os.path.join(path, os.listdir(path)[i+1]))

    #se estandariza el tamano de las imagenes para el analisis
    dim = (960, 540)
    img_1 = cv2.resize(img_1_o, dim, interpolation = cv2.INTER_AREA)
    img_2 = cv2.resize(img_2_o, dim, interpolation = cv2.INTER_AREA)

    #se concatenan las imagenes para mostrarse una al lado de la otra
    img = cv2.hconcat([img_1, img_2])

    #se generan vectores vacios para almacenar los puntos de referencia de manera temporal entre par de imagenes
    points = []
    points1 = []
    points2 = []
    point_counter = 0

    while True:
        cv2.imshow("Image", img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("x"):
            points = []
            break

        #se alterna el color de los puntos (rojo y azul) entre las dos imagenes a referenciar
        if len(points) > point_counter:
            point_counter = len(points)
            if point_counter % 2 == 0:
                cv2.circle(img, (points[-1][0], points[-1][1]), 3, [255, 0, 0], -1)
                points2.append(points[-1])
            else:
                cv2.circle(img, (points[-1][0], points[-1][1]), 3, [0, 0, 255], -1)
                points1.append(points[-1])

    #se verifica que se asignen al menos 4 puntos por imagen
    N = min(len(points1), len(points2))
    assert N >= 4, 'Se requieren al menos 4 puntos por imagen'
    points_g.append([points1, points2])

    #se guardan los puntos definidos por el usuario en cada imagen
    pts1 = np.array(points1[:N])
    pts2 = np.array(points2[:N])
    #print(pts2)
    pts2 -= (img_1.shape[1], 0)
    #print(pts2)

    #se genera la homografia entre imagenes dados los puntos definidos
    if False:
         H, _ = cv2.findHomography(pts1, pts2, method=0)
    else:
         H, _ = cv2.findHomography(pts1, pts2, method=cv2.RANSAC)
    H_t.append(H)
    cv2.waitKey(key)

#se cierran las imagenes analizadas
cv2. destroyAllWindows()
st = []


#### Homografia ####
for i in range(N_p):
    print(i)
    #se encuentra la homografia a la imagen de referencia definida por el usuario, considerando las transformaciones necesarias para las
    #imagenes a la izquierda de esta
    if i < img_central -1:
        j = i
        img = cv2.imread(os.path.join(path, os.listdir(path)[i]))
        while j < img_central - 1:
            img = cv2.warpPerspective(img, H_t[j], (img.shape[1], img.shape[0]))
            j += 1

    #se encuentra la homografia a la imagen de referencia definida por el usuario, considerando las transformaciones e inversas necesarias para las
    #imagenes a la derecha de esta
    elif i > img_central - 1:
        j = i -1
        img = cv2.imread(os.path.join(path, os.listdir(path)[i]))
        while j >= img_central - 1:
            img = cv2.warpPerspective(img, np.linalg.inv(H_t[j]), (img.shape[1], img.shape[0]))
            j -= 1

    else:
        img = cv2.imread(os.path.join(path, os.listdir(path)[i]))

    #se guardan las imagenes resultantes
    st.append(img)
print(len(st))

#se toma la imagene de referencia definida por el usuario
avg_image = st[img_central - 1]

#para las imagenes resultantes se excluyen los pixeles que quedaaron en negro
for x in range(avg_image.shape[0]):
    for y in range(avg_image.shape[1]):
        for p in range(len(st)):
            if p == img_central - 1:
                pass
            if st[p][x, y, 0] == 0 and st[p][x, y, 1] == 0 and st[p][x, y, 2] == 0:
                st[p][x, y, 0] = avg_image[x, y, 0]
                st[p][x, y, 1] = avg_image[x, y, 1]
                st[p][x, y, 2] = avg_image[x, y, 2]


#se promedian las imagenes resulatntes
for i in range(len(st)):
    if i == img_central -1 :
        pass
    else:
        alpha = 1.0/(i + 1)
        beta = 1.0 - alpha
        avg_image = cv2.addWeighted(st[i], alpha, avg_image, beta, 0.0)

#se guarda la imagen final
cv2.imwrite('Stitched_image.png', avg_image)
#se visualiza la imagen final
cv2.imshow('Imagen final', avg_image)
cv2.waitKey(0)
cv2.destroyWindow()


