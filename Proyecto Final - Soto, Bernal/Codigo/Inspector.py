import cv2
import numpy as np
import pandas as pd

import os
import re
import imutils
import matplotlib.pyplot as plt
import sys

'''
Clase Inspector -->>> Contiene los metodos necesarios para hacer la comparacion de fotogramas contra una serie de imagenes de referencia
'''
class Inspector():
    def __Init__(self, path_ob, path_ref):
        self.path_o = path_ob
        self.path_r = path_ref

#### Metodos para leer la carpeta de archivos y ordenar las imagenes numericamente 01..02..03.......n
    def atoi(self, text):
        return int(text) if text.isdigit() else text

    def natural_keys(self, text):
        '''
        alist.sort(key=natural_keys) sorts in human order
        http://nedbatchelder.com/blog/200712/human_sorting.html
        (See Toothy's implementation in the comments)
        '''
        return [self.atoi(c) for c in re.split(r'(\d+)', text)]

## Metodo para leer los archivos dada una ruta
    def Call_folder(self, folder):
        imgs = []
        for images in os.listdir(folder):
            imgs.append(images)
        imgs.sort(key=self.natural_keys)
        # print(imgs[2:32])
        return imgs[2:32]

### Se recorta la imagen para eliminar el fondo de la escena
    def Crop(self, folder, show = False):
        # Crop
        scale = 100

        p1 = [int(415 * scale / 50), int(155 * scale / 50)]
        p2 = [int(490 * scale / 50), int(500 * scale / 50)]

        imgs = self.Call_folder(folder)
        images_St = []
        for image in imgs:
            img = cv2.imread(os.path.join(folder, image))
            img = img[p1[1]:p2[1], p1[0]:p2[0]]

            images_St.append(img)
        return images_St

### Metodo para calcular la diferencia entre imagenes usando el MSE
    def mse(self, imageA, imageB):
        # the 'Mean Squared Error' between the two images is the
        # sum of the squared difference between the two images;
        # NOTE: the two images must have the same dimension
        #imageA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
        #imageB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
        err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
        err /= float(imageA.shape[0] * imageA.shape[1])

        # return the MSE, the lower the error, the more "similar"
        # the two images are
        return err

### Metodo para mostrar las imagenes contenidas en una lista
    def Show_can(self, images, t = 1000):
        if isinstance(images, list):
            if len(images) > 1:
                res = [x for x in images if x is not None]
            else:
                res = images
            if len(res) > 0:
                for image in res:
                    cv2.imshow('imagen', image)
                    cv2.waitKey(t)
            else:
                print('No hay imagenes para mostrar')
        else:
            if images is None:
                print('No hay imagenes para mostrar')
            else:
                cv2.imshow('imagen', images)
                cv2.waitKey(t)

        ''' Metodo para hacer la comparacion de imagenes
            imagenes_St --> imagenes de referencia
            imagenes_obj --> imagenes a inspeccionar
            show --> True para mostrar las imagenes de todo el proceso de inspeccion
            frame --> Imagen a inspeccionar (hace parte del listado de images_obj
            show_class --> Muestra unicamente las imagenes del proceso de deteccion de diferencias 
        '''
    def Inspect(self, images_St, images_obj, show = False, frame = 14, show_clas = False):
        i = frame
        scores = []
        for image_b in images_St:
            score = self.mse(image_b, images_obj[i])
            scores.append(score)

        # dada una imagen que se quiere inspeccionar, el metodo usa el MSE para encontrar la imagen mas parecida dentro del set de referencia
        best_score = np.argmin(scores)
        # print(f'Menor diferencia de {scores[best_score]} en la posicion {best_score}')
        error_ini = scores[best_score]
        if show:
            cv2.imshow('imagen', images_St[best_score])
            cv2.imshow('imageb', images_obj[i])
            cv2.waitKey(0)

    ###################################################
    
        # Se transforman imagenes a grises
        img_1 = images_St[best_score]
        img_2 = images_obj[i]
        image_gray_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
        image_gray_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)

        # Se calculan los Keypoints y se selecciona el metodo que mejor reduzca el error
        def best_method(metodo, show_b = show):
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

            if len(matches) < number_matches:
                number_matches = len(matches)

            # Se dibujan matches sobre las imagenes

            image_matching = cv2.drawMatches(img_1, keypoints_1, img_2, keypoints_2, matches[:number_matches], None)
            if show_b:
                cv2.imshow('Correspondencias Descriptores', image_matching)
                cv2.waitKey(0)

            # Se cargan los keypoints en contenedores
            points_1 = []
            points_2 = []
            for j in range(number_matches):
                idx = matches[j].queryIdx
                idx2 = matches[j].trainIdx
                points_1.append(keypoints_1[idx].pt)
                points_2.append(keypoints_2[idx2].pt)

            points1 = np.array(points_1)
            points2 = np.array(points_2)


            pts1 = []
            pts2 = []

            for k in range(len(points1)):

                if abs(points1[k][1] - points2[k][1]) < 2:
                    pts1.append(points1[k])
                    pts2.append(points2[k])

                else:
                    pass

            pts1 = np.array(pts1)
            pts2 = np.array(pts2)


            # Calculo de homografias para poder comparar de mejor manera las imagenes
            if len(pts1) > 4:
                H, _ = cv2.findHomography(pts1, pts2, method=cv2.RANSAC)

                #### Transformaciones --  Homografia ####

                img = img_2

                img = cv2.warpPerspective(img, np.linalg.inv(H), (img.shape[1], img.shape[0]))

                if show_b:
                    # print(img.shape)
                    cv2.imshow('Homograf Der', img)
                    cv2.waitKey(0)

                # Se carga la imagen central al contenedor de imagenes transformadas st

                # Union de imagenes
                avg_image = img_1

                # filtrando puntos negros de la homografia para que coincidan con la imagen de referencia


                for x in range(avg_image.shape[0]):
                    for y in range(avg_image.shape[1]):
                        if img[x, y, 0] == 0 and img[x, y, 1] == 0 and img[x, y, 2] == 0:
                            img[x, y, 0] = avg_image[x, y, 0]
                            img[x, y, 1] = avg_image[x, y, 1]
                            img[x, y, 2] = avg_image[x, y, 2]
                if show_b:
                    # print(img.shape)
                    cv2.imshow('Homograf rell', img)
                    cv2.waitKey(0)

                ##################################################
                error = self.mse(img_1, img)
                # print(f'Error: {error}')
                return error, img
            else:
                return 1e6, img_2

        err_1, im_1 = best_method(metodo = 1, show_b = show)
        err_2, im_2 = best_method(metodo = 2, show_b = show)

        if err_1 <= error_ini and err_2 <= error_ini:
            if err_1 < err_2:
                img = im_1
            else:
                img = im_2
        else:
            return None
        if show:
            cv2.imshow('im1', im_1)
            cv2.imshow('im2', im_2)
            cv2.imshow('im_final', img)
            cv2.waitKey(0)




        ## Se aplica threshold a las imagenes para reducir el ruido y calcular las diferencias
        im1g = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
        imgg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        (T, threshInv) = cv2.threshold(im1g, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        (T, threshInv2) = cv2.threshold(imgg, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        ret,im1_t = cv2.threshold(im1g,150,255,cv2.THRESH_BINARY)
        ret,im_t = cv2.threshold(imgg,150,255,cv2.THRESH_BINARY)

        diff = im1_t.copy()
        cv2.absdiff(im1_t, im_t, diff)

        # Se encuentran diferencias
        diff = im1_t.copy()
        cv2.absdiff(threshInv, threshInv2, diff)

        # Erosion inicial de las diferencias seguidas de dilataciones para resaltar las desviaciones encontradas
        procc = diff.copy()
        procc = cv2.erode(procc, None, iterations= 1)
        for i in range(0, 3):
            kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
            procc = cv2.erode(procc, kernel, iterations=i + 1)
            if show_clas:
                cv2.imshow('Imagen Referencia', img_1)
                cv2.imshow('Process_or', img)
                cv2.imshow('Process_er', procc)
            procc = cv2.dilate(procc, None, iterations=i + 1)
            if show_clas:
                cv2.imshow('Process', procc)
                cv2.waitKey(500)
        eroded = procc

        # Ultima dilatacion para aumentar el tamanio de las desviaciones encontradas
        for i in range(0, 3):
            dilated = cv2.dilate(eroded.copy(), None, iterations=i + 1)



        if show:
            cv2.imshow('Diferencias', diff)
            cv2.imshow('Diferencias_Dilated', dilated)
            cv2.imshow('Diferencias_eroded', eroded)
            cv2.waitKey(0)

        ################################################
        # Se resaltan los contornos usando cajas
        edged = cv2.Canny(dilated, 30, 200)
        contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        try: hierarchy = hierarchy[0]
        except: hierarchy = []

        height, width = dilated.shape
        min_x, min_y = width, height
        max_x = max_y = 0

        for contour, hier in zip(contours, hierarchy):
            (x,y,w,h) = cv2.boundingRect(contour)
            min_x, max_x = min(x, min_x), max(x+w, max_x)
            min_y, max_y = min(y, min_y), max(y+h, max_y)
            if w > 80 and h > 80:
                cv2.rectangle(img_2, (x,y), (x+w,y+h), (0, 0, 255), 2)

        ## Dibujado de cajas
        if max_x - min_x > 10 and max_y - min_y > 10:
            cv2.rectangle(img, (int(min_x - min_x*0.1), int(min_y - min_y*0.1)), (int(max_x + max_x*0.1), int(max_y + max_y*0.1)), (0, 0, 255), 2)
            print('imagen adicionada')
            return img

            # if show:
            #     cv2.imshow('Deteccion', img)
            #     cv2.waitKey(0)
        else:
            print('No hay hallazgos')
            return None

        ##########################################################################################
