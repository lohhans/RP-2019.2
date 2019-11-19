import os
from skimage.io import imread
from skimage import io
from matplotlib import pyplot as plt

################# Importar dados Cup #################
caminhosCup = os.listdir("Dados/cup/")
imagensCup = []
for k in range(len(caminhosCup)):

    imagensCup.append(imread("Dados/cup/"+caminhosCup[k], True))

################# Importar dados Car #################
caminhosCar = os.listdir("Dados/car_side/")
imagensCar = []
for k in range(len(caminhosCar)):

    imagensCar.append(imread("Dados/car_side/"+caminhosCar[k], True))


################# Mostrar #################
#Cup
io.imshow(imagensCup[1])
plt.show()
#Car
io.imshow(imagensCar[1])
plt.show()
