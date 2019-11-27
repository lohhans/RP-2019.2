import os
import random
import numpy as np

from skimage import *
from skimage.io import imread
from skimage import io
from matplotlib import pyplot as plt
from skimage import feature
from skimage import filters
from skimage import morphology
from skimage.transform import rescale, resize

import torch
from torch.autograd import Variable
from torch.nn import Linear, ReLU, Sigmoid, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from torch.optim import Adam, SGD


from sklearn.metrics import confusion_matrix

################# Importar dados Garfield #################
caminhosGar = os.listdir("Dados/garfield/")
imagensGar = []
classeGar =[]
for k in range(len(caminhosGar)):

    imagensGar.append(imread("Dados/garfield/"+caminhosGar[k], True))
    classeGar.append("Garfield")

################# Importar dados Yin yang #################
caminhosYY = os.listdir("Dados/yin_yang/")
imagensYY = []
classeYY =[]
for k in range(len(caminhosYY)):

    imagensYY.append(imread("Dados/yin_yang/"+caminhosYY[k], True))
    classeYY.append("Yin yang")

random.shuffle(imagensYY) #embaralhar dados
random.shuffle(imagensGar) #embaralhar dados

imagens = imagensYY + imagensGar #MLP
imagensCNN = imagensYY + imagensGar #CNN

for k in range(len(imagens)):
#redimensionando as imagens
    imagens[k] = resize(imagens[k], (160, 160))
    imagensCNN[k] = resize(imagensCNN[k], (160, 160))


####################################################################
################    Manipulação das     ############################
################        imagens         ############################
####################################################################

for k in range(len(imagens)):

#redimensionando as imagens e aplicando  suavização

    imagens[k] = filters.median(imagens[k])

#segmentação
    imagens[k] = feature.canny(imagens[k])

    imagens[k] = util.img_as_float32(imagens[k])
#modelagem matemática
    imagens[k] = morphology.closing(imagens[k])

#extração de formas
    caracteristicas = []
    #pegar distancia linha 1
    temp1 = None
    temp2 = None
    for j in range(160):
        if imagens[k][39][j] == 1:
            if temp1 == None:
                temp1 = j
            else:
                temp2 = j
    if temp1 == None:
        temp1 = 159
    if temp2 == None:
        temp2 = 159

    caracteristicas.append(temp2 - temp1)

    #pegar distancia linha 2
    temp1 = None
    temp2 = None
    for j in range(160):
        if imagens[k][79][j] == 1:
            if temp1 == None:
                temp1 = j
            else:
                temp2 = j
    if temp1 == None:
        temp1 = 159
    if temp2 == None:
        temp2 = 159

    caracteristicas.append(temp2 - temp1)

    #pegar distancia linha 3
    temp1 = None
    temp2 = None
    for j in range(160):
        if imagens[k][119][j] == 1:
            if temp1 == None:
                temp1 = j
            else:
                temp2 = j
    if temp1 == None:
        temp1 = 159
    if temp2 == None:
        temp2 = 159

    caracteristicas.append(temp2 - temp1)

    #pegar distancia coluna 1
    temp1 = None
    temp2 = None
    for j in range(160):
        if imagens[k][j][39] == 1:
            if temp1 == None:
                temp1 = j
            else:
                temp2 = j
    if temp1 == None:
        temp1 = 159
    if temp2 == None:
        temp2 = 159

    caracteristicas.append(temp2 - temp1)

    #pegar distancia coluna 2
    temp1 = None
    temp2 = None
    for j in range(160):
        if imagens[k][j][79] == 1:
            if temp1 == None:
                temp1 = j
            else:
                temp2 = j
    if temp1 == None:
        temp1 = 159
    if temp2 == None:
        temp2 = 159

    caracteristicas.append(temp2 - temp1)

    #pegar distancia coluna 3
    temp1 = None
    temp2 = None
    for j in range(160):
        if imagens[k][j][119] == 1:
            if temp1 == None:
                temp1 = j
            else:
                temp2 = j
    if temp1 == None:
        temp1 = 159
    if temp2 == None:
        temp2 = 159

    caracteristicas.append(temp2 - temp1)
    imagens[k] = caracteristicas

#separando dados em treino, teste e validação
treino = []
classTreino = []

for k in range(37):
    treino.append(imagens[k])
    classTreino.append([1,0])
for k in range(60, 80):
    treino.append(imagens[k])
    classTreino.append([0,1])

teste = []
classTeste = []

for k in range(37, 55):
    teste.append(imagens[k])
    classTeste.append([1,0])
for k in range(80, 90):
    teste.append(imagens[k])
    classTeste.append([0,1])

validacao = []
classValidacao = []

for k in range(55, 60):
    validacao.append(imagens[k])
    classValidacao.append([1,0])
for k in range(90, len(imagens)):
    validacao.append(imagens[k])
    classValidacao.append([0,1])



####################################################################
################           Rede         ############################
################           MLP          ############################
####################################################################

#convertendo dados para formato Tensor
treino = torch.autograd.Variable(torch.FloatTensor(treino))
classTreino = torch.autograd.Variable(torch.FloatTensor(classTreino))
teste = torch.autograd.Variable(torch.FloatTensor(teste))
classTeste = torch.autograd.Variable(torch.FloatTensor(classTeste))
validacao = torch.autograd.Variable(torch.FloatTensor(validacao))
classValidacao = torch.autograd.Variable(torch.FloatTensor(classValidacao))

#modelo da rede
class Modelo(torch.nn.Module):
        def __init__(self, tamanho):
            super(Modelo, self).__init__()
            self.camada1 = torch.nn.Linear(tamanho, 4)
            self.camada2 = torch.nn.Linear(4, 2)
            self.sigmoid = torch.nn.Sigmoid()

        def forward(self, entrada):
            #primeira camada
            camada1 = self.camada1(entrada)
            camada1 = self.sigmoid(camada1)

            #segunda camada
            camada2 = self.camada2(camada1)
            camada2 = self.sigmoid(camada2)

            return camada2

#treinando a MLP
modelo = Modelo(6)
criterion = torch.nn.BCELoss()  #taxa de erro
lr = torch.optim.SGD(modelo.parameters(), lr = 0.01)    #definindo learning rate

#ver taxa de erro antes do treino
modelo.eval()
y_pred = modelo.forward(treino)
before_train = criterion(y_pred.squeeze(), classTreino)
print('Antes do treino: ', before_train.item(), "\n\n")

#iniciando treino
modelo.train()
epoch = 100000 # 100000 épocas

train_losses = []
val_losses = []

for epoch in range(epoch):

    lr.zero_grad()
    #colocando dados na rede
    y_pred = modelo.forward(treino)
    y_val = modelo.forward(validacao)

    #calculando a perca
    loss = criterion(y_pred.squeeze(), classTreino)
    loss_val = criterion(y_val.squeeze(), classValidacao)

    #armazenando perdas
    train_losses.append(((-loss)+1))
    val_losses.append(((-loss_val)+1))

    loss.backward()
    lr.step()

#testando modelo após o treino
modelo.eval()
modelo = modelo.forward(teste)
after_train = criterion(modelo.squeeze(), classTeste)
print('Teste depois do treino' , after_train.item())
modelo = modelo.detach()
modelo = modelo.numpy()

classTeste = classTeste.detach()
classTeste = classTeste.numpy()

resultado = []
for k in range(len(modelo)):
    if(modelo[k][0]>modelo[k][1]):
        resultado.append(1)
    else:
        resultado.append(0)

classTeste2 = []
for k in range(len(classTeste)):
    if(classTeste[k][0]>classTeste[k][1]):
        classTeste2.append(1)
    else:
        classTeste2.append(0)
print("Matriz de confusao MLP:")
print(confusion_matrix(classTeste2, resultado)
)

#mostrar gráfico de perdas
plt.plot(train_losses, label='Acerto do treino')
plt.plot(val_losses, label='Acerto da validação')
plt.legend()
plt.show()

####################################################################
################           Rede         ############################
################           CNN          ############################
####################################################################

#manipulando dados
treinoCNN = []
classTreinoCNN = []

#criando vetores de classificação
for k in range(37):
    treinoCNN.append(imagensCNN[k])
    classTreinoCNN.append(1)
for k in range(60, 80):
    treinoCNN.append(imagensCNN[k])
    classTreinoCNN.append(0)

testeCNN = []
classTesteCNN = []

for k in range(37, 55):
    testeCNN.append(imagensCNN[k])
    classTesteCNN.append(1)
for k in range(80, 90):
    testeCNN.append(imagensCNN[k])
    classTesteCNN.append(0)

validacaoCNN = []
classValidacaoCNN = []

for k in range(55, 60):
    validacaoCNN.append(imagensCNN[k])
    classValidacaoCNN.append(1)
for k in range(90, len(imagensCNN)):
    validacaoCNN.append(imagensCNN[k])
    classValidacaoCNN.append(0)

#convertendo dados para Tensor
treinoCNN = torch.autograd.Variable(torch.FloatTensor(treinoCNN))

classTreinoCNN = np.array(classTreinoCNN)
classTreinoCNN = classTreinoCNN.astype(float)
classTreinoCNN = torch.from_numpy(classTreinoCNN)
classTreinoCNN = classTreinoCNN.long()


testeCNN = torch.autograd.Variable(torch.FloatTensor(testeCNN))

classTesteCNN = np.array(classTesteCNN)
classTesteCNN = np.array(classTesteCNN)
classTesteCNN = classTesteCNN.astype(float)
classTesteCNN = torch.from_numpy(classTesteCNN)
classTesteCNN = classTesteCNN.long()

validacaoCNN = torch.autograd.Variable(torch.FloatTensor(validacaoCNN))

classValidacaoCNN = np.array(classValidacaoCNN)
classValidacaoCNN = np.array(classValidacaoCNN)
classValidacaoCNN = np.array(classValidacaoCNN)
classValidacaoCNN = classValidacaoCNN.astype(float)
classValidacaoCNN = torch.from_numpy(classValidacaoCNN)
classValidacaoCNN = classValidacaoCNN.long()


#modelo da rede CNN

class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.cnn_layers = Sequential(
            #primeira camada convolucional
            Conv2d(1, 10, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(10),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),

            # Defining another 2D convolution layer
            Conv2d(10, 5, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(5),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),

            # Defining another 2D convolution layer
            Conv2d(5, 4, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(4),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
        )

        self.linear_layers = Sequential(
            #computação das características
            #camada 1
            Linear(1*40*40, 20),
            ReLU(),
            #camada 2
            Linear(20, 10),
            ReLU(),
            #camada 3
            Linear(10, 5),
            ReLU(),
            #camada 4
            Linear(5, 4),
            Sigmoid(),
            #camada 5
            Linear(4, 2),
            Sigmoid(),
        )

    def forward(self, x):
        #aplica as camadas
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        #joga as imagens no classificador
        x = self.linear_layers(x)
        return x

validacaoCNN = validacaoCNN.reshape(9, 1, 160, 160)

model = CNN()
optimizer = Adam(model.parameters(), lr=0.07)
criterion = CrossEntropyLoss()
# checking if GPU is available
if torch.cuda.is_available():
    model = model.cuda()
    criterion = criterion.cuda()

treinoCNN = treinoCNN.reshape(57, 1, 160, 160)
validacaoCNN = validacaoCNN.reshape(9, 1, 160, 160)

############### Treino CNN ##################


model = CNN()
optimizer = Adam(model.parameters(), lr=0.001)
criterion = CrossEntropyLoss()
if torch.cuda.is_available():
    model = model.cuda()
    criterion = criterion.cuda()

def train(epoch, treinoCNN, validacaoCNN, classTreinoCNN, classValidacaoCNN):
    model.train()
    tr_loss = 0

    if torch.cuda.is_available():

        treinoCNN = treinoCNN.cuda()
        classTreinoCNN = classTreinoCNN.cuda()
        validacaoCNN = validacaoCNN.cuda()
        classValidacaoCNN = classValidacaoCNN.cuda()

    optimizer.zero_grad()
    output_train = model(treinoCNN)
    output_val = model(validacaoCNN)

    #calculando o ganho
    loss_train = criterion(output_train, classTreinoCNN)
    loss_val = criterion(output_val, classValidacaoCNN)
    train_losses.append(((-loss_train)+1))
    val_losses.append(((-loss_val)+1))


    loss_train.backward()
    optimizer.step()
    tr_loss = loss_train.item()
    # if epoch%2 == 0:
        # printing the validation loss
        # print('Epoch : ',epoch+1, '\t', 'loss :', loss_val)

n_epochs = 100
train_losses = []
val_losses = []

#treinando a rede
for epoch in range(n_epochs):
    train(epoch, treinoCNN, validacaoCNN, classTreinoCNN, classValidacaoCNN)


# plotting the training and validation loss
plt.plot(train_losses, label='Acerto do treino')
plt.plot(val_losses, label='Acerto da validação')
plt.legend()
plt.show()

########################################################

model.eval()
testeCNN = testeCNN.reshape(28, 1, 160, 160)
modelo = model(testeCNN)
print(modelo)
modelo = modelo.detach()
modelo = modelo.numpy()
print(modelo)

resultado = []

for k in range(len(modelo)):
    if(modelo[k][0]>modelo[k][1]):
        resultado.append(1)
    else:
        resultado.append(0)

print("Matriz de confusão CNN:")
print(confusion_matrix(classTesteCNN, resultado)
)
