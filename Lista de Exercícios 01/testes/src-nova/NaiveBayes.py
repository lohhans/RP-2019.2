import os
import pandas as pd

from Dados import *
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB

# TESTES!!!

'''
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

'''

# Instanciando arquivos para listas
script_dir = os.path.dirname(__file__) # Diretório absoluto
rp1 = "../testes/data/peace.csv"              # Diretório relativo
rp2 = "../testes/data/silence.csv"            # Diretório relativo
rp3 = "../testes/data/success.csv"            # Diretório relativo
fp1 = os.path.join(script_dir, rp1)    # Diretório final
fp2 = os.path.join(script_dir, rp2)    # Diretório final
fp3 = os.path.join(script_dir, rp3)    # Diretório final

dd = Dados()
# dd.importarCSV(fp1, fp2, fp3)
# print(dd.treino)

dadosPanda = pd.read_csv(fp1, sep='\t', encoding='utf-8')
print(dadosPanda.describe())

print('\n')

print(dadosPanda)

print('\n')

print(dadosPanda.columns)

print('\n\n\nignorar\n\n\n')

'''
x = CountVectorizer()

# transforma treinoFrase em uma matriz de numeros por palavra
treinoFrase = x.fit_transform(treinoFrase).toarray()

# cria a bagOfWords
bagOfWords = x.vocabulary_

y_vect = CountVectorizer(vocabulary=bagOfWords)  # adiciona a bagOfWords a nova variavel
bagOfWords = y_vect.fit_transform(testeFrase).toarray()  # coloca todas as frases na nova bagOfWords

# Inicializa o classificador
gnb = GaussianNB()

# Treina o classificador
nb = gnb.fit(treinoFrase, treinoClass)

# mostra score
print("A precisão é de", str(nb.score(bagOfWords, testeClass) * 100) + "%")

#Antonio testando

print(gnb.predict(bagOfWords))
'''
