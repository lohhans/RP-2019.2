import random
from sklearn.feature_extraction.text import CountVectorizer

class Dados:

    def __init__(self):
        self.treino = []
        self.teste = []

    # Antônio ↴

    def importarDados(self, aqv1, aqv2, aqv3, nome1, nome2, nome3):
        dados = self.pegarAqv(aqv1, nome1) + self.pegarAqv(aqv2, nome2) + self.pegarAqv(aqv3, nome3)
        random.shuffle(dados)

        for k in range(270):
            self.treino.append(dados[k])  # COLOCA 60% DOS DADOS PARA TREINO

        for k in range(270, len(dados)):
            self.teste.append(dados[k])  # COLOCA 40# DOS DADOS PARA TESTE

    def pegarAqv(self, nomeDoArquivo, nomeClasse):

        arquivo = open(nomeDoArquivo, 'r')
        frases = arquivo.readlines()  # LENDO TODAS AS FRASES DO ARQUIVO
        classe = []

        for k in range(len(frases)):  # REMOVENDO OS \N's DAS STR
            # COLOCANDO AS STR NA LIST DE DADOS
            frases[k] = frases[k].rstrip('\n')
            # COLOCANDO AS CLASSES NA LIST DE CLASSES
            classe.append(nomeClasse)

        arquivo.close()

        dados = list(zip(frases, classe))  # JUNTA AS LISTAS EM UMA

        return dados

    # Métodos para o formato CSV
    # Armstrong ↴

    def gerarLista(self, nomeDoArquivo):

        arquivo = open(nomeDoArquivo, 'r')
        frases = arquivo.readlines()  # LENDO TODAS AS FRASES DO ARQUIVO
        tupla = []

        for k in range(len(frases)):  # REMOVENDO OS \N's DAS STR
            # COLOCANDO AS STR NA LIST DE DADOS
            frases[k] = frases[k].rstrip('\n')
            # COLOCANDO AS CLASSES NA LIST DE CLASSES
            # tupla.append(nomeClasse)

        arquivo.close()

        # dados = list(zip(frases, tupla))  # JUNTA AS LISTAS EM UMA

        return frases

    def importarCSV(self, aqv1, aqv2, aqv3):
        dados = self.gerarLista(aqv1) + self.gerarLista(aqv2) + self.gerarLista(aqv3)
        random.shuffle(dados)

        for k in range(270):
            self.treino.append(dados[k])  # COLOCA 60% DOS DADOS PARA TREINO

        for k in range(270, len(dados)):
            self.teste.append(dados[k])  # COLOCA 40# DOS DADOS PARA TESTE

    # Antônio ↴

    def importarDadosMaiores(self, aqv1, aqv2, aqv3, nome1, nome2, nome3):
        dados = self.pegarAqv(aqv1, nome1) + self.pegarAqv(aqv2, nome2) + self.pegarAqv(aqv3, nome3)
        dadosFrases = []
        dadosClass = []

        for k in range(len(dados)):
            dadosFrases.append(dados[k][0])
            dadosClass.append(dados[k][1])

        x = CountVectorizer()
        treinoFrase = x.fit_transform(dadosFrases).toarray()# transforma treinoFrase em uma matriz de numeros por palavra
        dadosRetorno = []

        dadosRetorno.append(treinoFrase)
        dadosRetorno.append(dadosClass)

        return dadosRetorno
