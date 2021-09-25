from Quadrilateral import Quadrilateral

import cv2
from hough import Hough
import numpy as np

# Definicion de tamano de imagen
N = 650

# Se crea objeto de la clase Quadrilateral
q1 = Quadrilateral(N)

# Se genera imagen
img = q1.generate()

'''
La funcion DetectCorners recibe una imagen en RGB de un poligono con maximo 10 lados y retorna la imagen con las esquinas 
resaltadas dentro de una circunferencia amarilla y un arreglo con las posiciones de estas.
'''

def DetectCorners(image):
    # Deteccion de bordes
    high_thresh = 200
    bw_edges = cv2.Canny(image, high_thresh * 0.3, high_thresh, L2gradient=True)

    # Validacion del numero de lados del poligono
    cnts = cv2.findContours(bw_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.03 * peri, True)
        n_ed = len(approx)

    # se valida condicion de numero de lados menor o igual que 10
    if n_ed <= 10:
        # Uso de la transformada de Hough Standar
        hough = Hough(bw_edges)

        accumulator = hough.standard_transform()
        acc_thresh = 50
        N_peaks = n_ed  # Se limita el numero de lineas a la cantidad de lados detectados
        nhood = [25, 9]
        peaks = hough.find_peaks(accumulator, nhood, acc_thresh, N_peaks)


        _, cols = image.shape[:2]
        image_draw = np.copy(image)

        # Identificacion de los puntos que definen las lineas encontradas
        lines = []
        for peak in peaks:
            rho = peak[0]
            theta_ = hough.theta[peak[1]]

            theta_pi = np.pi * theta_ / 180
            theta_ = theta_ - 180
            a = np.cos(theta_pi)
            b = np.sin(theta_pi)
            x0 = a * rho + hough.center_x
            y0 = b * rho + hough.center_y
            c = -rho
            x1 = int(round(x0 + cols * (-b)))
            y1 = int(round(y0 + cols * a))
            x2 = int(round(x0 - cols * (-b)))
            y2 = int(round(y0 - cols * a))

            p1 = (x1, y1)
            p2 = (x2, y2)
            image_draw = cv2.line(image_draw, (x1, y1), (x2, y2), [0, 0, 0], thickness=2)

            lines.append([p1, p2]) # Se almacenan en un contenedor

        # Inteseccion de lineas

        def line_intersection(line1, line2):
            xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
            ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

            def det(a, b):
                return a[0] * b[1] - a[1] * b[0]

            div = det(xdiff, ydiff)
            if div == 0:
                pass
            else:
                d = (det(*line1), det(*line2))
                x = det(d, xdiff) / div
                y = det(d, ydiff) / div
                return x, y

        # Se cruzan las lineas detectadas para identificar los puntos de interseccion
        points = []
        for line_1 in lines:
            for line_2 in lines:
                if line_1 == line_2:
                    pass
                else:
                    points.append(line_intersection(line_1, line_2))

        points = list(set(points)) # Se eliminan puntos duplicados

        # Se eliminan valores Nulos
        res = []
        for val in points:
            if val != None:
                res.append([int(val[0]), int(val[1])])


        # Dibujado de las circunferencias amarillas sobre los vertices del poligono
        for p in res:
            image = cv2.circle(image, (p[0], p[1]), 10, (0, 233, 255), 2)

        # Generacion del arreglo de los puntos de interseccion - vertices
        corners = np.array(res)


        return image, corners

    else:
        print('Numero de lados mayor a 10')


# se corre el metodo
img, corners = DetectCorners(img)

print(f'Posiciones de las esquinas: \n {corners}')

cv2.imshow('Inter', img)
cv2.waitKey(0)


# cv2.imshow('imagen', img)
# cv2.waitKey(0)