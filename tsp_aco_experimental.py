#!/usr/bin/python3
# -*- coding: utf-8 -*-

# Rodrigo Mansilha
# Universidade Federal do Pampa (Unipampa)
# Programa de Pós-Graduação em Eng. de Software (PPGES)
# Bacharelado em: Ciência da Camputação, Eng. de Software, Eng. de Telecomunicações

# Algoritmos
# Laboratório 2: avaliação experimental

from tsp_aco import bubblesort


DEFAULT_SEED = None
DEFAULT_N_START = 1
DEFAULT_N_STOP = 10       
DEFAULT_N_STEP = 1
DEFAULT_TRIALS = 3
DEFAULT_N_MAX  = None
DEFAULT_OUTPUT =  "tsp_aco_experimental.png"
DEFAULT_ALGORITMOS = None # executa todos

try:
	import sys
	import os
	import argparse
	import logging
	import subprocess
	import shlex
	from abc import ABC, abstractmethod

	from scipy.special import factorial

	import math
	import numpy as np
	import matplotlib.pyplot as plt
	import scipy.optimize as opt
	import matplotlib.colors as colors
	import matplotlib.cm as cmx


except ImportError as error:
	print(error)
	print()
	print("1. (optional) Setup a virtual environment: ")
	print("  python3 -m venv ~/Python3env/algoritmos ")
	print("  source ~/Python3env/algoritmos/bin/activate ")
	print()
	print("2. Install requirements:")
	print("  pip3 install --upgrade pip")
	print("  pip3 install -r requirements.txt ")
	print()
	sys.exit(-1)

# Lista completa de mapas de cores
# https://matplotlib.org/examples/color/colormaps_reference.html
mapa_cor = plt.get_cmap('tab20')  # carrega tabela de cores conforme dicionário
mapeamento_normalizado = colors.Normalize(vmin=0, vmax=19)  # mapeamento em 20 cores
mapa_escalar = cmx.ScalarMappable(norm=mapeamento_normalizado, cmap=mapa_cor)  # lista de cores final

formatos = ['.', 'v', 'o', '^', '<', '>', '1', '2', '3', '4', 's', 'p', '*', 'h']


# https://matplotlib.org/3.3.2/api/_as_gen/matplotlib.pyplot.plot.html
# '.'	point marker
# ','	pixel marker
# 'o'	circle marker
# 'v'	triangle_down marker
# '^'	triangle_up marker
# '<'	triangle_left marker
# '>'	triangle_right marker
# '1'	tri_down marker
# '2'	tri_up marker
# '3'	tri_left marker
# '4'	tri_right marker
# 's'	square marker
# 'p'	pentagon marker
# '*'	star marker
# 'h'	hexagon1 marker
# 'H'	hexagon2 marker
# '+'	plus marker
# 'x'	x marker
# 'D'	diamond marker
# 'd'	thin_diamond marker
# '|'	vline marker
# '_'	hline marker


def funcao_fatorial(n, cpu):
	'''
	Aproximação fatorial
	:param n: tamanho da instância
	:param cpu: fator de conversão para tempo de CPU
	:return: aproximação
	'''
	return (factorial(n) * cpu)


def funcao_quadratica(n, cpu):
	'''
	Aproximação quadrática
	:param n: tamanho da instância
	:param cpu: fator de conversão para tempo de CPU
	:return: aproximação
	'''
	return (n * n * cpu)



def funcao_linear(n, cpu):
	'''
	Aproximação linear
	:param n: tamanho da instância
	:param cpu: fator de conversão para tempo de CPU
	:return: aproximação
	'''
	return (n * cpu)


def imprime_config(args):
	'''
	Mostra os argumentos recebidos e as configurações processadas
	:args: parser.parse_args
	'''
	print("Argumentos:\n\t{0}\n".format(" ".join([x for x in sys.argv])))
	print("Configurações:")
	for k, v in sorted(vars(args).items()): print("\t{0}: {1}".format(k, v))
	print("")


class Experimento(ABC):

	@abstractmethod
	def __init__(self, args):
		self.id = "letraid"
		self.args = args
		self.script = "abc.py"
		self.output = "out.txt"
		self.tamanhos = []
		self.medias = []
		self.desvios = []
		self.aproximados = []

		#configurações de plotagem
		self.medicao_legenda = "medicao_legenda"
		self.medicao_cor_rgb = mapa_escalar.to_rgba(0)
		self.medicao_formato = formatos[1]

		self.aproximacao_legenda = "aproximacao_legenda"
		self.aproximacao_cor_rgb = mapa_escalar.to_rgba(1)

	def executa_experimentos(self):
		# cria string de comando
		comando_str = "python3 {}".format(self.script)
		comando_str += " --out {}".format(self.output)
		comando_str += " --nstart {}".format(self.args.nstart)
		comando_str += " --nstop {}".format(self.args.nstop)
		comando_str += " --nstep {}".format(self.args.nstep)
		comando_str += " --trials {}".format(self.args.trials)
		if self.args.seed is not None:
			comando_str += " --seed {}".format(self.args.seed)

		print("Comando: {}".format(comando_str))
		# transforma em array por questões de segurança -> https://docs.python.org/3/library/shlex.html
		comando_array = shlex.split(comando_str)

		# executa comando em subprocesso
		subprocess.run(comando_array)

	def carrega_resultados(self):
		'''
		Carrega dados do arquivo de saída para a memória principal
		'''
		f = open(self.output, "r")

		for l in f:
			print("linha: {}".format(l))
			if l[0] != "#":
				self.tamanhos.append(int(l.split(" ")[0]))
				self.medias.append(float(l.split(" ")[1]))
				self.desvios.append(float(l.split(" ")[2]))
		f.close()

	def imprime_dados(self):
		# mostra dados
		print("Tamanho\tMedia\t\tDesvio\t\tAproximado")
		for i in range(len(self.tamanhos)):
			print("%03d\t%02f\t%02f\t%02f" % (self.tamanhos[i], self.medias[i], self.desvios[i], self.aproximados[i]))
		print("")

	@abstractmethod
	def executa_aproximacao(self):
		pass

	def plota_aproximacao(self):
		plt.plot(self.tamanhos, self.aproximados, label=self.aproximacao_legenda, color=self.aproximacao_cor_rgb)

	def plota_medicao(self):
		plt.errorbar(x=self.tamanhos, y=self.medias, yerr=self.desvios, fmt=self.medicao_formato,
					 label=self.medicao_legenda, color=self.medicao_cor_rgb, linewidth=2)




class TSP_ACO(Experimento):

	def __init__(self, args):
		super().__init__(args)
		self.id = "s"
		self.script = "tsp_aco.py"
		self.output = "tsp_aco.txt"

	    # configurações de plotagem
		self.medicao_legenda = "tsp_aco medido"
		self.medicao_cor_rgb = mapa_escalar.to_rgba(2)
		self.medicao_formato = formatos[3]

		self.aproximacao_legenda = "tsp_aco aproximado"
		self.aproximacao_cor_rgb = mapa_escalar.to_rgba(3)

	def executa_aproximacao(self):
		# realiza aproximação
		parametros, pcov = opt.curve_fit(funcao_quadratica, xdata=self.tamanhos, ydata=self.medias)
		self.aproximados = [funcao_quadratica(x, *parametros) for x in self.tamanhos]
		print("aproximados:           {}".format(self.aproximados))
		print("parametros_otimizados: {}".format(parametros))
		print("pcov:                  {}".format(pcov))
        


def main():
	'''
	Programa principal
	:return:
	'''

	# Definição de argumentos
	parser = argparse.ArgumentParser(description='Laboratório 2')

	help_msg = "semente aleatória"
	parser.add_argument("--seed", "-s", help=help_msg, default=DEFAULT_SEED, type=int)

	help_msg = "n máximo.          Padrão:{}".format(DEFAULT_N_STOP)
	parser.add_argument("--nstop", "-n", help=help_msg, default=DEFAULT_N_STOP, type=int)

	help_msg = "n mínimo.          Padrão:{}".format(DEFAULT_N_START)
	parser.add_argument("--nstart", "-a", help=help_msg, default=DEFAULT_N_START, type=int)

	help_msg = "n passo.           Padrão:{}".format(DEFAULT_N_STEP)
	parser.add_argument("--nstep", "-e", help=help_msg, default=DEFAULT_N_STEP, type=int)

	help_msg = "n máximo.          Padrão:{}".format(DEFAULT_N_MAX)
	parser.add_argument("--nmax", "-m", help=help_msg, default=DEFAULT_N_MAX, type=int)

	help_msg = "tentativas.        Padrão:{}".format(DEFAULT_TRIALS)
	parser.add_argument("--trials", "-t", help=help_msg, default=DEFAULT_TRIALS, type=int)

	help_msg = "figura (extensão .png ou .pdf) ou nenhum para apresentar na tela.  Padrão:{}".format(DEFAULT_OUTPUT)
	parser.add_argument("--out", "-o", help=help_msg, default=DEFAULT_OUTPUT, type=str)

	help_msg = "algoritmos (t=tsp naive, s=selection sort) ou nenhum para executar todos.  Padrão:{}".format(DEFAULT_ALGORITMOS)
	parser.add_argument("--algoritmos", "-l", help=help_msg, default=DEFAULT_ALGORITMOS, type=str)

	# Lê argumentos da linha de comando
	args = parser.parse_args()

	# imprime configurações para fins de log
	imprime_config(args)

	# lista de experimentos disponíveis
	experimentos = [TSP_ACO(args)]

	for e in experimentos:
		if args.algoritmos is None or e.id in args.algoritmos:
			e.executa_experimentos()
			e.carrega_resultados()
			e.executa_aproximacao()
			e.imprime_dados()
			e.plota_medicao()
			e.plota_aproximacao()

	# configurações gerais
	plt.legend()
	plt.xticks(range(args.nstart, args.nstop+1, args.nstep))
	plt.title("Impacto da otimização das rotas - TSP_ACO - Daniel")
	plt.xlabel("Tamanho da instância (n)")
	plt.ylabel("Tempo de execução (s)")

	if args.out is None:
		# mostra
		plt.show()
	else:
		# salva em png
		plt.savefig(args.out, dpi=300)


if __name__ == '__main__':
	sys.exit(main())
