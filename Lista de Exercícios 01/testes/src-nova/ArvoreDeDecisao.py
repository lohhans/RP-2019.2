import os
import graphviz

import numpy as np
from Dados import *
from sklearn.tree import _tree
from sklearn import preprocessing, tree
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score


# Instanciando arquivos para listas
script_dir = os.path.dirname(__file__)                  # Diretório absoluto
rel_path1 = "../testes/data/peace.txt"                       # Diretório relativo
rel_path2 = "../testes/data/silence.txt"                     # Diretório relativo
rel_path3 = "../testes/data/success.txt"                     # Diretório relativo
abs_file_path1 = os.path.join(script_dir, rel_path1)    # Diretório final
abs_file_path2 = os.path.join(script_dir, rel_path2)    # Diretório final
abs_file_path3 = os.path.join(script_dir, rel_path3)    # Diretório final

script = os.path.dirname(__file__)                  # Diretório absoluto
path1 = "../frasesMais/PeaceMais.txt"                       # Diretório relativo
path2 = "../frasesMais/SilenceMais.txt"                     # Diretório relativo
path3 = "../frasesMais/SuccessMais.txt"                     # Diretório relativo
file_path1 = os.path.join(script, path1)    # Diretório final
file_path2 = os.path.join(script, path2)    # Diretório final
file_path3 = os.path.join(script, path3)    # Diretório final

# instanciando Listas de dados
treinoFrase = []
treinoClass = []

testeFrase = []
testeClass = []

# instanciando classe de dados
dados = Dados()
dados.importarDados(abs_file_path1, abs_file_path2, abs_file_path1, "Peace", "Silence", "Success")

# pegando os dados e separando nas listas instanciadas acima
for k in range(len(dados.treino)):
    treinoFrase.append(dados.treino[k][0])
    treinoClass.append(dados.treino[k][1])

for k in range(len(dados.teste)):
    testeFrase.append(dados.teste[k][0])
    testeClass.append(dados.teste[k][1])

x = CountVectorizer()

# transforma treinoFrase em uma matriz de numeros por palavra
treinoFrase = x.fit_transform(treinoFrase).toarray()

# cria a bagOfWords
bagOfWords = x.vocabulary_

y_vect = CountVectorizer(vocabulary=bagOfWords)  # adiciona a bagOfWords a nova variavel
bagOfWords = y_vect.fit_transform(testeFrase).toarray()  # coloca todas as frases na nova bagOfWords

arvore = tree.DecisionTreeClassifier()  # cria arvore
arvore.fit(treinoFrase, treinoClass)  # treina arvore
# mostra score
print("A precisão é: ", str(arvore.score(bagOfWords, testeClass) * 100) + "%")
# classifica a frase
frase = [str(input("Digite a frase: "))]    #recebe a frase e coloca em um vetor
fraseMatriz = x.transform(frase).toarray()  #transforma a frase em vertor
print("A classificação é: ",str(arvore.predict(fraseMatriz)))   #infere a classificacao

#print("O caminho é: ", str(arvore.decision_path(fraseMatriz)))

dot_data = tree.export_graphviz(arvore)
graph = graphviz.Source(dot_data)
graph.render("arvoreImagem")


#validacao cruzada

#pegar dados
dadosMais = dados.importarDadosMaiores(file_path1, file_path2, file_path1, "Peace", "Silence", "Success")
#validacao
cruzada = cross_val_score(arvore, dadosMais[0], dadosMais[1], cv=5) #calcular as validacoes
cruzada = (cruzada[0] + cruzada[1] + cruzada[2] + cruzada[3] + cruzada[4])/5 #calcular medias
print("A média de precisão por validação cruzada é: ", str(cruzada * 100) + "%")
