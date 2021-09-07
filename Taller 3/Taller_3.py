import cv2
import numpy as np
import os
import sys

""" 
Taller 3 Alisson Bernal, Juan Nicolas Soto
Funciones para diezmado, Interpolado y filtrado de imagenes
"""


#path_file = r'C:\Users\juann\Downloads/lena.png'
#path_file = r'C:\Users\juann\PycharmProjects\pythonProject1\Proce_imag\Talleres\Taller_2\01_1.tif'
#image = cv2.imread(path_file)

#####################################
##### Diezmado ######################

'''
La funcion Decimante (Diezmado) recibe tres argumentos:
1. Imagen en varios canales o Blanco y negro.
2. El factor D de diezmado
3. Una variable booleanan show que imprime en pantalla las imagenes originales, transformada FFT, mascara, FFT con mascara e imagen diezmada. Por defecto se setea en False.
Retorna la imagen diezmada 
'''
def Decimate(image_file, D=2, show = False):
    # Se valida que el parametro D sea entero mayor que 1
    if D <= 1 or D%1 != 0:
        print('El parametro D debe ser un entero mayor que 1')
        return
    img = image_file

    # Si la imagen tiene mas de un canal se transforma a grises
    if len(img.shape) > 2:
        image_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        image_gray = img

    # Transformando la imagen a FFT
    image_gray_fft = np.fft.fft2(image_gray)
    image_gray_fft_shift = np.fft.fftshift(image_gray_fft)

    # Visualización FFT
    image_gray_fft_mag = np.absolute(image_gray_fft_shift)
    image_fft_view = np.log(image_gray_fft_mag + 1)
    image_fft_view = image_fft_view / np.max(image_fft_view)

    #### Filtrado #####
    filter_mask = np.zeros_like(image_fft_view)  # Mascara de ceros
    half_x = int(filter_mask.shape[0] / 2)      # Centros de la imagen
    half_y = int(filter_mask.shape[1] / 2)      # Centros de la imagen

    # Se define un radio igual a la frecuencia de corte half_y/D.A todos los puntos dentro de ese radio se les asigna el valor de 1
    for i in range(filter_mask.shape[0]):
        for j in range(filter_mask.shape[0]):
            if np.sqrt((i-half_y)**2 + (j-half_x)**2) <= half_y/D:
                filter_mask[i][j] = 1
    filter_mask[half_x][half_y] =1

    # Visualización de la mascara
    filter_mask_view = filter_mask*255

    # Aplicación de la mascara a la FFT y posterior reconversión para ser visualizada
    filter_image_fft = filter_mask * image_fft_view
    fft_filtered = image_gray_fft_shift * filter_mask
    image_filtered = np.fft.ifft2(np.fft.fftshift(fft_filtered))
    image_filtered = np.absolute(image_filtered)
    image_filtered -= np.min(image_filtered)
    image_filtered /= np.max(image_filtered)

    # Diezmado de la imagen filtrada
    image_filtered = image_filtered[::D, ::D]

    ######### FFT de Imagen Diezmada##########
    image_filtered_fft = np.fft.fft2(image_filtered)
    image_filtered_fft_shift = np.fft.fftshift(image_filtered_fft)

    # fft visualization
    image_gray_fft_mag_filtered = np.absolute(image_filtered_fft_shift)
    image_fft_view_fil = np.log(image_gray_fft_mag_filtered + 1)
    image_fft_view_fil = image_fft_view_fil / np.max(image_fft_view_fil)
    ############################################################

    # Visualización de todas las etapas del proceso de diezmado.
    if show:
        cv2.imshow('Imgray', image_gray)
        cv2.imshow('filter', filter_mask_view)
        cv2.imshow('Imfilter', image_fft_view)
        cv2.imshow('ImfilterMask', filter_image_fft)
        cv2.imshow('ImDeci', image_filtered)
        cv2.imshow('ImDeciFFT', image_fft_view_fil)
        cv2.waitKey(0)

    # Imagen Retornada
    return image_filtered

#########################
##### Interpolado ##########
'''
La funcion Interpolated (Interpolado) recibe tres argumentos:
1. Imagen en varios canales o Blanco y negro.
2. El factor D de Interpolado
3. Una variable booleanan show que imprime en pantalla las imagenes originales, transformada FFT, mascara, FFT con mascara e imagen diezmada. Por defecto se setea en False.
Retorna la imagen Interpolada 
'''
def Interpolate(image_file, I=2, show = False):
    # Se valida que el parametro I sea entero mayor que 1
    if I <= 1 or I%1 != 0:
        print('El parametro I debe ser un entero mayor que 1')
        return
    img = image_file

    # Si la imagen tiene mas de un canal se transforma a grises
    if len(img.shape) > 2:
        image_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        image_gray = img

    # Inserción de ceros
    rows, cols = image_gray.shape
    num_of_zeros = I
    image_zeros = np.zeros((num_of_zeros * rows, num_of_zeros * cols), dtype=image_gray.dtype)
    image_zeros[::num_of_zeros, ::num_of_zeros] = image_gray

    ############# Filtrado ###################
    image_gray_fft = np.fft.fft2(image_zeros)
    image_gray_fft_shift = np.fft.fftshift(image_gray_fft)

    # Visualización FFT
    image_gray_fft_mag = np.absolute(image_gray_fft_shift)
    image_fft_view = np.log(image_gray_fft_mag + 1)
    image_fft_view = image_fft_view / np.max(image_fft_view)

    #### Mascara de Filtrado #####
    filter_mask = np.zeros_like(image_fft_view) # Mascara de ceros
    half_x = int(filter_mask.shape[0] / 2)      # Centro de la imagen
    half_y = int(filter_mask.shape[1] / 2)      # Centro de la imagen

    # Se define un radio igual a la frecuencia de corte half_y/D.A todos los puntos dentro de ese radio se les asigna el valor de 1
    for i in range(filter_mask.shape[0]):
        for j in range(filter_mask.shape[0]):
            if np.sqrt((i-half_y)**2 + (j-half_x)**2) <= half_y/I:
                filter_mask[i][j] = 1
    filter_mask[half_x][half_y] =1

    # Visualización de la mascara
    filter_mask_view = filter_mask*255

    # Aplicación de la mascara a la FFT y posterior reconversión para ser visualizada
    filter_image_fft = filter_mask * image_fft_view
    fft_filtered = image_gray_fft_shift * filter_mask
    image_filtered = np.fft.ifft2(np.fft.fftshift(fft_filtered))
    image_filtered = np.absolute(image_filtered)
    image_filtered -= np.min(image_filtered)
    image_filtered /= np.max(image_filtered)

    # Visualización de todas las etapas del proceso de Interpolado.
    if show:
        cv2.imshow('Imgray', image_gray)
        cv2.imshow('ImgZeros', image_zeros)
        cv2.imshow('ImgFFT', image_fft_view)
        cv2.imshow('ImgFilteMask', filter_mask_view)
        cv2.imshow('ImgInterp', image_filtered)
        cv2.waitKey(0)

    # Retorna imagen Interpolada
    return image_filtered


#########################
##### Banco de filtrado ##########
'''
La funcion Filters (Banco de fitrado) recibe dos argumentos:
1. Imagen en varios canales o Blanco y negro.
3. Una variable booleanan show que imprime en pantalla las imagenes originales, transformada FFT, mascara, FFT con mascara e imagen diezmada. Por defecto se setea en False.
Retorna la imagen Interpolada 
'''

def Filters(image_file, show = False):
    lista_img = [] # Contenedor de imagenes filtradas
    img = image_file

    # Conversión a grises
    image_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Definición de los filtros
    H = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    V = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    D = np.array([[2, -1, -2], [-1, 4, -1], [-2, -1, 2]])
    L = np.array([[1/16, 1/8, 1/16], [1/8, 1/4, 1/8], [1/16, 1/8, 1/16]])

    # Primera etapa de filtrado
    con_l1 = cv2.filter2D(src=image_gray, ddepth=-1, kernel=L)
    con_h1 = cv2.filter2D(src=image_gray, ddepth=-1, kernel=H)
    con_v1 = cv2.filter2D(src=image_gray, ddepth=-1, kernel=V)
    con_d1 = cv2.filter2D(src=image_gray, ddepth=-1, kernel=D)

    # Diezmado de las imagenes filtradas
    IH = Decimate(con_h1, 2)
    IV = Decimate(con_v1, 2)
    ID = Decimate(con_d1, 2)
    IL = Decimate(con_l1, 2)

    # Segunda etapa de filtrado
    con_l2 = cv2.filter2D(src=IL, ddepth=-1, kernel=L)
    con_h2 = cv2.filter2D(src=IL, ddepth=-1, kernel=H)
    con_v2 = cv2.filter2D(src=IL, ddepth=-1, kernel=V)
    con_d2 = cv2.filter2D(src=IL, ddepth=-1, kernel=D)

    # Diezmado de las imagenes filtradas
    ILH = Decimate(con_h2, 2)
    ILV = Decimate(con_v2, 2)
    ILD = Decimate(con_d2, 2)
    ILL = Decimate(con_l2, 2)

    # Se guardan imagenes y nombres en contenedores
    lista_img = [IH,IV,ID, ILH, ILV, ILD, ILL]
    lista_names = ['IH', 'IV', 'ID', 'ILH', 'ILV', 'ILD', 'ILL']

    # # Visualización de todas las etapas del proceso de Filtrado
    if show:
        cv2.imshow('Original', image_gray)
        cv2.imshow('IH', IH)
        cv2.imshow('IV', IV)
        cv2.imshow('ID', ID)
        cv2.imshow('IL', IL)
        cv2.imshow('ILH', ILH)
        cv2.imshow('ILV', ILV)
        cv2.imshow('ILD', ILD)
        cv2.imshow('ILL', ILL)
        cv2.imshow('ILL', ILL_inter)
        cv2.waitKey(0)

    # Retorna Lista de imagenes y lista de nombres
    return lista_img, lista_names



