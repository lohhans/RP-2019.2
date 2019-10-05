import os
from Dados import *
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score

# Instanciando arquivos para listas
script_dir = os.path.dirname(__file__)                  # Diretório absoluto
rel_path1 = "../frases/peace.txt"                       # Diretório relativo
rel_path2 = "../frases/silence.txt"                     # Diretório relativo
rel_path3 = "../frases/success.txt"                     # Diretório relativo
abs_file_path1 = os.path.join(script_dir, rel_path1)    # Diretório final
abs_file_path2 = os.path.join(script_dir, rel_path2)    # Diretório final
abs_file_path3 = os.path.join(script_dir, rel_path3)    # Diretório final

script = os.path.dirname(__file__)          # Diretório absoluto
path1 = "../frasesMais/PeaceMais.txt"       # Diretório relativo
path2 = "../frasesMais/SilenceMais.txt"     # Diretório relativo
path3 = "../frasesMais/SuccessMais.txt"     # Diretório relativo
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

# Inicializa o classificador
gnb = GaussianNB()

# Treina o classificador
nb = gnb.fit(treinoFrase, treinoClass)

# Mostra score
print("A precisão é de", str(nb.score(bagOfWords, testeClass) * 100) + "%\n")

# Mostra as a Posteriori de cada classe
classes = gnb.classes_
aPosteriori = gnb.class_prior_

for k in range(len(classes)):
    print("Classe: ", classes[k], " a posteriori: ", aPosteriori[k])

#Validacao cruzada
#Pegar dados
dadosMais = dados.importarDadosMaiores(file_path1, file_path2, file_path1, "Peace", "Silence", "Success")
#Validacao
cruzada = cross_val_score(gnb, dadosMais[0], dadosMais[1], cv=5) #Calcular as validacoes
cruzada = (cruzada[0] + cruzada[1] + cruzada[2] + cruzada[3] + cruzada[4])/5 #Calcular medias
print("\nA média de precisão por validação cruzada é: ", str(cruzada * 100) + "%")
