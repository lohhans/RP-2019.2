import random


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

    # Métodos simples para gerar uma lista do TXT
    # Armstrong ↴

    def listaSimples(self, nomeDoArquivo):

        arquivo = open(nomeDoArquivo, 'r')
        frases = arquivo.readlines()
        lista = []

        for k in range(len(frases)):
            lista.append(frases[k].rstrip('\n'))

        arquivo.close()

        return lista

    def importarDadosSimples(self, aqv1, aqv2, aqv3):
        dados = self.listaSimples(aqv1) + self.listaSimples(aqv2) + self.listaSimples(aqv3)
        random.shuffle(dados)

        for k in range(270):
            self.treino.append(dados[k])  # Coloca 60% dos dados para treino

        for k in range(270, len(dados)):
            self.teste.append(dados[k])  # Coloca 40# dos dados para teste
