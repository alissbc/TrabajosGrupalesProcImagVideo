import cv2
# Se importa la clase Inspector
from Inspector import Inspector

# Se crea objeto
Can = Inspector()

# Se leen las imagenes de referencia y objetivo
folder_r = r'D:\Maestria IA\Proyecto_inspector\Fotos_Latas\Club_N2'
folder_o = r'D:\Maestria IA\Proyecto_inspector\Fotos_Latas\Club_N2_c1'
img_st = Can.Crop(folder_r)
img_obj = Can.Crop(folder_o)

#Can.Show_can(img_obj, t = 200)

result = []

# Inspeccion de las imagenes contenidas en objetivo
# for i in range(len(img_obj)):
#     print(i)
#     inspection = Can.Inspect(img_st, img_obj, show = False, frame = i, show_clas= True)
#     result.append(inspection)

result = Can.Inspect(img_st, img_obj, show = True, frame = 14, show_clas=True)

# Se muestran los hallazgos
Can.Show_can(result, t = 0)




