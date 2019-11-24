import os
from Dados import *


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB

from sklearn.tree import _tree
from sklearn import preprocessing, tree
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

# Instanciando arquivos para listas
script_dir = os.path.dirname(__file__)                  # Diretório absoluto
rel_path1 = "../frases/peace.txt"                       # Diretório relativo
rel_path2 = "../frases/silence.txt"                     # Diretório relativo
rel_path3 = "../frases/success.txt"                     # Diretório relativo
abs_file_path1 = os.path.join(script_dir, rel_path1)    # Diretório final
abs_file_path2 = os.path.join(script_dir, rel_path2)    # Diretório final
abs_file_path3 = os.path.join(script_dir, rel_path3)    # Diretório final

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

#Cria o Naive
gnb = GaussianNB()
nb = gnb.fit(treinoFrase, treinoClass)
#Cria a Arvore
arvore = tree.DecisionTreeClassifier()
arvore.fit(treinoFrase, treinoClass)
print("\n##################################################################################")
print("################################### LISTA 1 ######################################");
print("##################################################################################\n\n")

frase = [str(input("Digite a frase: "))]    #recebe a frase e coloca em um vetor
fraseMatriz = x.transform(frase).toarray()  #transforma a frase em vertor

if (nb.score(bagOfWords, testeClass) > arvore.score(bagOfWords, testeClass)):
    print("A melhor classificação é: ",str(gnb.predict(fraseMatriz)[0]))
    classificacao = str(gnb.predict(fraseMatriz)[0])
else:
    print("A melhor classificação é: ",str(arvore.predict(fraseMatriz)[0]))
    classificacao = str(arvore.predict(fraseMatriz)[0])

classes = list(gnb.classes_)
aPosteriori = gnb.class_prior_

print("A probabilidade a posteriori foi de:", str(aPosteriori[classes.index(classificacao)]*100) + "%")
print("\n\n A regra de decisão foi:\n",str(arvore.decision_path(fraseMatriz)))
print("\nVocê pode ver a árvore de decisão completa na pasta raiz")
print("\n##################################################################################")

numero = int(input("\nDigite 1 para ver a base de teste : "))
if (numero == 1):
    for k in range(len(testeFrase)):
        fraseMatriz = x.transform([testeFrase[k]]).toarray()  #transforma a frase em vertor
        print("\n##################################################################################")
        if (nb.score(bagOfWords, testeClass) > arvore.score(bagOfWords, testeClass)):
            print("\n\nA melhor classificação é: ",str(gnb.predict(fraseMatriz)[0]))
            classificacao = str(gnb.predict(fraseMatriz)[0])
        else:
            print("\n\nA melhor classificação é: ",str(arvore.predict(fraseMatriz)[0]))
            classificacao = str(arvore.predict(fraseMatriz)[0])

        classes = list(gnb.classes_)
        aPosteriori = gnb.class_prior_

        print("\nA probailidade a prosteriori foi de :", str(aPosteriori[classes.index(classificacao)]*100) + "%")
        print("\n A regra de decisão foi:\n",str(arvore.decision_path(fraseMatriz)))
        print("\n##################################################################################")
