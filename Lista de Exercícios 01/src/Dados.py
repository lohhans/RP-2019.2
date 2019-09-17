class Dados:

    def __init__(self):
        self.dados = []
        self.classes = []


    def importarDados(self, nomeDoArquivo, nomeClasse):

        arquivo = open(nomeDoArquivo, 'r')
        frases = arquivo.readlines()                        #LENDO TODAS AS FRASES DO ARQUIVO

        for k in range(len(frases)):                        #REMOVENDO OS \N's DAS STR
            frases[k] = frases[k].rstrip('\n')

            self.dados.append(frases[k])                    #COLOCANDO AS STR NA LIST DE DADOS
            self.classes.append(nomeClasse)                 #COLOCANDO AS CLASSES NA LIST DE CLASSES

        arquivo.close()

#######################################################################################################################
                                        #TESTES#
#######################################################################################################################

# d = Dados()
# d.importarDados('/home/antonio/Documentos/RP-2019.2/Lista de Exercícios 01/frases/funny.txt', 'teste')
#
# print(d.dados)
# print(d.classes)
