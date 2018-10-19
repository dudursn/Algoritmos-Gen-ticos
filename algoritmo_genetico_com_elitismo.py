'''
Algoritmo genetico feito por Eduardo Roger Silva Nascimento

'''
import numpy as np
from math import pi, exp, sqrt, cos
import matplotlib.pyplot as plt


MIN = -32.768
MAX = 32.768
#Criterio de parada
MAX_GEN = 1000
TAM_POP = 100
TAM_CROM = 64
TX_CROSS = 0.9
TX_MUT = 0.001
N =2

#Gera uma população de 100 elementos com 64 bits, onde 32 codifica x1 e 32 codifica x2
def gera_populacao():
    return np.random.randint(0,2, size=(TAM_POP, TAM_CROM))

#Faz a decodificação de binario para o decimal normalizado na faixa (-32,768; 32,768)
def decodificacao(cromossomo):
	#Pega x1
	x1_bin = cromossomo[:32]
	#Pega x2
	x2_bin = cromossomo[32:]
	#Faz a normalização
	x1 = normalizacao(bin_para_int(x1_bin))
	x2 = normalizacao(bin_para_int(x2_bin))
	return [x1, x2]

#Converte de binario para decimal
def bin_para_int(numeroBin):
    return int(''.join(str(x) for x in numeroBin),2)

#Normaliza na faixa (-32,768; 32,768)
def normalizacao(decimal):
	return MIN + ((MAX - MIN)*(decimal/(2**32-1)))

#Faz o cálculo da aptidão
def aptidao(cromossomo):
	#x é o vetor
    x = decodificacao(cromossomo)
    #Função  a ser maximizada
    return 1/(1+f(x))

#Retorna a aptidão de cada individuo da população
def calcula_aptidao(pop):
    return [aptidao(i) for i in pop]

def f(x):
	return -20*exp(-0.2*sqrt((1/N)*somatorio1(x)))-exp((1/N)*somatorio2(x))+20+exp(1)

def somatorio1(x):
	soma = 0
	for i in range(N):
		soma += x[i]**2
	return soma

def somatorio2(x):
	soma = 0
	for i in range(N):
		soma += cos(2*pi*x[i])
	return soma

def selecao_roleta(aptidoes):
    percentuais = np.array(aptidoes)/float(sum(aptidoes))
    vet = [percentuais[0]]
    for p in percentuais[1:]:
        vet.append(vet[-1]+p)
    r = np.random.random()
    for i in range(len(vet)):
        if r <= vet[i]:
            return i
            
def cruzamento(pai,mae):
    corte = np.random.randint(TAM_CROM)
    return (list(pai[:corte])+list(mae[corte:]),list(mae[:corte])+list(pai[corte:]))

def mutacao(cromossomo):
    r = np.random.randint(TAM_CROM)
    cromossomo[r] = (cromossomo[r]+1)%2
    return cromossomo


def algoritmo_genetico():
	#Cria a população
	populacao = gera_populacao()

	medias = []
	desvio_padrao = []
	melhores = []

	for geracao in range(MAX_GEN):
		#Inicia com o cálculo das aptidões da população 
		aptidoes = calcula_aptidao(populacao)

		nova_populacao = []
		
		for c in range(int(TAM_POP/2)):
			'''
			print(len(populacao))
			print(len(aptidoes))
			'''
			#Seleção
			pai = populacao[selecao_roleta(aptidoes)]
			mae = populacao[selecao_roleta(aptidoes)]

			#Cruzamento
			r = np.random.random()
			if r <= TX_CROSS:
				filho,filha = cruzamento(pai,mae)
			else:
				filho,filha = pai,mae

			#Mutação
			r = np.random.random()
			if r <= TX_MUT:
				filho = mutacao(filho)
			r = np.random.random()
			if r <= TX_MUT:
				filha = mutacao(filha)

			#Nova população sendo criada
			nova_populacao.append(filho)
			nova_populacao.append(filha)

		#Fitness medio
		medias.append(np.mean(aptidoes))

		#Desvio padrão
		desvio_padrao.append(np.std(aptidoes))

		#Melhores
		melhores.append(np.max(aptidoes))

		#Imprime a última média calculada
		print(geracao+1,'-',medias[-1], desvio_padrao[-1])			


		#Elitismo
		#Busca-se o índice que contém o menor valor entre as aptidões da nova população, ou seja possivel pior filho
		novas_aptidoes = calcula_aptidao(nova_populacao)
		index_pior_filho = novas_aptidoes.index(min(novas_aptidoes))
		#Busca-se o índice que contém o mair valor entre as aptidões da população anterior, ou seja melhor pai
		index_melhor_pai = aptidoes.index(max(aptidoes))
		#Apaga o pior da nova geração
		nova_populacao = np.delete(nova_populacao, index_pior_filho, 0)
		#Adiciona o melhor da geração anterior a nova geração
		nova_populacao = np.insert(nova_populacao, index_pior_filho, populacao[index_melhor_pai],0)

		#A população atual passa a ser a nova geração
		populacao = nova_populacao


		

	#Assim que acaba a última iteração, calcula-se a aptidão da última geração
	aptidoes = calcula_aptidao(populacao)

	#Busca-se o índice que contém o maior valor entre as aptidões
	index_solucao = aptidoes.index(max(aptidoes))
	
	print("Resposta final: " + str(decodificacao(populacao[index_solucao])))

	#Plotando o gráfico
	print("Gerando gráficos...")
	plt.figure(1)
	plt.subplot(211)
	
	plt.ylabel('Médias')
	plt.plot([x for x in range(1, MAX_GEN+1)], [medias[i] for i in range(len(medias))], 'ro')
	plt.axis([0, MAX_GEN+1, 0, max(medias)])

	plt.subplot(212)
	plt.xlabel('Geração')
	plt.ylabel('Aptidões')
	plt.plot([x for x in range(1, MAX_GEN+1)], [melhores[i] for i in range(len(melhores))], 'g^')
	plt.axis([0, MAX_GEN+1, 0, max(melhores)])
	plt.show()

algoritmo_genetico()