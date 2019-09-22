import os

import pandas as pd
from Dados import *
from sklearn import preprocessing

# Instanciando arquivos para listas
script_dir = os.path.dirname(__file__)  # Diretório absoluto
rel_path1 = "../frases/peace.txt"  # Diretório relativo
rel_path2 = "../frases/silence.txt"  # Diretório relativo
rel_path3 = "../frases/success.txt"  # Diretório relativo
abs_file_path1 = os.path.join(script_dir, rel_path1)  # Diretório final
abs_file_path2 = os.path.join(script_dir, rel_path2)  # Diretório final
abs_file_path3 = os.path.join(script_dir, rel_path3)  # Diretório final

dados = Dados()

dados.importarDadosSimples(abs_file_path1, abs_file_path2, abs_file_path1)

print(dados.treino)
print('\n')
print(dados.teste)

print('\n')

# Criando labelEncoder
le = preprocessing.LabelEncoder()

# Convertendo rótulos de string em números
frases_encoded = le.fit_transform(dados.teste)

print(frases_encoded)
