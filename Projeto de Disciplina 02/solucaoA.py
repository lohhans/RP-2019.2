import os
from skimage.io import imread
from skimage import io
from matplotlib import pyplot as plt

from skimage import feature
from skimage import filters
from skimage import morphology

import numpy as np
import torch
from skimage.filters import threshold_otsu

from skimage import *
from skimage.transform import rescale, resize, downscale_local_mean
from skimage.measure import find_contours, approximate_polygon, subdivide_polygon

################# Importar dados accordion #################
caminhosGar = os.listdir("Dados/garfield/")
imagensGar = []
for k in range(len(caminhosGar)):

    imagensGar.append(imread("Dados/garfield/"+caminhosGar[k], True))


################# Importar dados yin yang #################
caminhosYY = os.listdir("Dados/yin_yang/")
imagensYY = []
for k in range(len(caminhosYY)):

    imagensYY.append(imread("Dados/yin_yang/"+caminhosYY[k], True))

####################################################################
#########################Acc########################################
####################################################################

for k in range(len(imagensGar)):


    imagensGar[k] = resize(imagensGar[k], (160, 160))
    imagensGar[k] = filters.median(imagensGar[k])


#segmentação
    imagensGar[k] = feature.canny(imagensGar[k])
    imagensGar[k] = util.img_as_float32(imagensGar[k])
#modelagem matemática
    imagensGar[k] = morphology.closing(imagensGar[k])
#extração de formas
    imagensGar[k] = util.img_as_float32(imagensGar[k])
    #imagensGar[k] = morphology.flood(imagensGar[k], (0,0))

    # print(imagensGar[k])
    #imagensGar[k] = approximate_polygon(imagensGar[k],  1 )



####################################################################
#########################Y_Y########################################
####################################################################

for k in range(len(imagensYY)):


    imagensYY[k] = resize(imagensYY[k], (160, 160))
    imagensYY[k] = filters.median(imagensYY[k])


#segmentação
    imagensYY[k] = feature.canny(imagensYY[k])
#modelagem matemática
    imagensYY[k] = morphology.closing(imagensYY[k])
#extração de formas
    imagensYY[k] = util.img_as_float32(imagensYY[k])
    #imagensYY[k] = morphology.flood(imagensYY[k], (0,0))

###################### MLP ###########################


class Feedforward(torch.nn.Module):
        def __init__(self, input_size, hidden_size):
            super(Feedforward, self).__init__()
            self.input_size = input_size
            self.hidden_size  = hidden_size
            self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
            self.relu = torch.nn.ReLU()
            self.fc2 = torch.nn.Linear(self.hidden_size, 1)
            self.sigmoid = torch.nn.Sigmoid()
        def forward(self, x):
            hidden = self.fc1(x)
            relu = self.relu(hidden)
            output = self.fc2(relu)
            output = self.sigmoid(output)
            return output




################# Mostrar #################
#YY
#for k in range(len(imagensYY)):

# for k in range(len(imagensGar[0])):
# caracteristicas = []
#
# #pegar distancia linha 1
# temp1 = None
# temp2 = None
# for k in range(160):
#     if imagensYY[0][39][k] == 1:
#         if temp1 == None:
#             temp1 = k
#         else:
#             temp2 = k
# if temp1 == None:
#     temp1 = 159
# if temp2 == None:
#     temp2 = 159
#
# caracteristicas.append(temp2 - temp1)
#
# #pegar distancia linha 2
# temp1 = None
# temp2 = None
# for k in range(160):
#     if imagensYY[0][79][k] == 1:
#         if temp1 == None:
#             temp1 = k
#         else:
#             temp2 = k
# if temp1 == None:
#     temp1 = 159
# if temp2 == None:
#     temp2 = 159
#
# caracteristicas.append(temp2 - temp1)
#
# #pegar distancia linha 3
# temp1 = None
# temp2 = None
# for k in range(160):
#     if imagensYY[0][119][k] == 1:
#         if temp1 == None:
#             temp1 = k
#         else:
#             temp2 = k
# if temp1 == None:
#     temp1 = 159
# if temp2 == None:
#     temp2 = 159
#
# caracteristicas.append(temp2 - temp1)
#
# #pegar distancia coluna 1
# temp1 = None
# temp2 = None
# for k in range(160):
#     if imagensYY[0][k][39] == 1:
#         if temp1 == None:
#             temp1 = k
#         else:
#             temp2 = k
# if temp1 == None:
#     temp1 = 159
# if temp2 == None:
#     temp2 = 159
#
# caracteristicas.append(temp2 - temp1)
#
# #pegar distancia coluna 2
# temp1 = None
# temp2 = None
# for k in range(160):
#     if imagensYY[0][k][79] == 1:
#         if temp1 == None:
#             temp1 = k
#         else:
#             temp2 = k
# if temp1 == None:
#     temp1 = 159
# if temp2 == None:
#     temp2 = 159
#
# caracteristicas.append(temp2 - temp1)
#
# #pegar distancia coluna 3
# temp1 = None
# temp2 = None
# for k in range(160):
#     if imagensYY[0][k][119] == 1:
#         if temp1 == None:
#             temp1 = k
#         else:
#             temp2 = k
# if temp1 == None:
#     temp1 = 159
# if temp2 == None:
#     temp2 = 159
#
# caracteristicas.append(temp2 - temp1)







caracteristicas = []

#pegar distancia linha 1
temp1 = None
temp2 = None
for k in range(160):
    if imagensGar[0][39][k] == 1:
        if temp1 == None:
            temp1 = k
        else:
            temp2 = k
if temp1 == None:
    temp1 = 159
if temp2 == None:
    temp2 = 159

caracteristicas.append(temp2 - temp1)

#pegar distancia linha 2
temp1 = None
temp2 = None
for k in range(160):
    if imagensGar[0][79][k] == 1:
        if temp1 == None:
            temp1 = k
        else:
            temp2 = k
if temp1 == None:
    temp1 = 159
if temp2 == None:
    temp2 = 159

caracteristicas.append(temp2 - temp1)

#pegar distancia linha 3
temp1 = None
temp2 = None
for k in range(160):
    if imagensGar[0][119][k] == 1:
        if temp1 == None:
            temp1 = k
        else:
            temp2 = k
if temp1 == None:
    temp1 = 159
if temp2 == None:
    temp2 = 159

caracteristicas.append(temp2 - temp1)

#pegar distancia coluna 1
temp1 = None
temp2 = None
for k in range(160):
    if imagensGar[0][k][39] == 1:
        if temp1 == None:
            temp1 = k
        else:
            temp2 = k
if temp1 == None:
    temp1 = 159
if temp2 == None:
    temp2 = 159

caracteristicas.append(temp2 - temp1)

#pegar distancia coluna 2
temp1 = None
temp2 = None
for k in range(160):
    if imagensGar[0][k][79] == 1:
        if temp1 == None:
            temp1 = k
        else:
            temp2 = k
if temp1 == None:
    temp1 = 159
if temp2 == None:
    temp2 = 159

caracteristicas.append(temp2 - temp1)

#pegar distancia coluna 3
temp1 = None
temp2 = None
for k in range(160):
    if imagensGar[0][k][119] == 1:
        if temp1 == None:
            temp1 = k
        else:
            temp2 = k
if temp1 == None:
    temp1 = 159
if temp2 == None:
    temp2 = 159

caracteristicas.append(temp2 - temp1)

print(str(caracteristicas)+"\n")

io.imshow(imagensGar[0])
plt.show()
# for k in range(len(imagensYY)):
#     io.imshow(imagensYY[k])
#     plt.show()
#Garfield
# for k in range(len(imagensGar)):
#     io.imshow(imagensGar[k])
#     plt.show()

# for k in range(len(imagensGar[0])):
#     print(str(imagensGar[0][k])+"\n")
