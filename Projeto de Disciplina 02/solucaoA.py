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

################# Importar dados Cup #################
caminhosCup = os.listdir("Dados/cup/")
imagensCup = []
for k in range(len(caminhosCup)):

    imagensCup.append(imread("Dados/cup/"+caminhosCup[k], True))

imagemBit = []
for k in range(len(imagensCup)):
    imagemBit.append(np.zeros(imagensCup[k].shape))

################# Importar dados Car #################
caminhosCar = os.listdir("Dados/car_side/")
imagensCar = []
for k in range(len(caminhosCar)):

    imagensCar.append(imread("Dados/car_side/"+caminhosCar[k], True))

########################################################

for k in range(len(imagensCup)):


    imagensCup[k] = resize(imagensCup[k], (150, 150))
    imagensCup[k] = filters.median(imagensCup[k])


#segmentação
    imagensCup[k] = feature.canny(imagensCup[k])
#modelagem matemática
    imagensCup[k] = morphology.closing(imagensCup[k])
#extração de formas
    imagensCup[k] = util.img_as_float32(imagensCup[k])
    imagensCup[k] = morphology.flood(imagensCup[k], (0,0))

    imagensCup[k] = morphology.selem.disk(2,  imagensCup[k] )
    # print(imagensCup[k])
    #imagensCup[k] = approximate_polygon(imagensCup[k],  1 )



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
#Cup
for k in range(len(imagensCup)):
    io.imshow(imagensCup[k])
    plt.show()
#Car
# io.imshow(imagensCar[1])
# plt.show()
