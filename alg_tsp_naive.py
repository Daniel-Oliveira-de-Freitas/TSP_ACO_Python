#!/usr/bin/python3
# -*- coding: utf-8 -*-

DEFAULT_OUT = "out_alg_naive_tsp.txt"
DEFAULT_SEED = None

DEFAULT_N_START = 1
DEFAULT_N_STOP = 10
DEFAULT_N_STEP = 1
DEFAULT_TRIALS = 3

from subprocess import Popen, PIPE
from time import sleep, time
from multiprocessing import Process
import shlex
import json

import sys
import os
import argparse
import logging
import subprocess

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import matplotlib.colors as colors
import matplotlib.cm as cmx

import timeit



# -------------------------------------------------------------------------
# fonte: https://stackoverflow.com/questions/18651871/generating-a-map-graph-for-a-traveling-salesmanproblem-python
import math


def dist(a, b):
	return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def planar_graph_generator(n, m):
	points = [[np.random.randint(0,m), np.random.randint(0,m)] for i in range(n)]
	graph = [[dist( points[i], points[j] ) for i in range(n)] for j in range(n)]
	print("points: ", points)
	print("graph: ", graph)
	return graph

# -------------------------------------------------------------------------

# --------------------------------------------------------------------------------------
# fonte  https://www.geeksforgeeks.org/traveling-salesman-problem-tsp-implementation/
# Python3 program to solve the traveling salesman
# problem using a naive approach.

from sys import maxsize
from itertools import permutations


def travellingSalesmanProblem(graph, s):
	V = len(graph[0])
	# store all vertex apart from source vertex
	vertex = []
	for i in range(V):
		if i != s:
			vertex.append(i)

	# store minimum weight Hamiltonian Cycle
	min_path = maxsize
	next_permutation = permutations(vertex)
	for i in next_permutation:

		# store current Path weight(cost)
		current_pathweight = 0

		# compute current path weight
		k = s
		for j in i:
			current_pathweight += graph[k][j]
			k = j
		current_pathweight += graph[k][s]

		# update minimum
		min_path = min(min_path, current_pathweight)

	return min_path


# Driver Code
def teste():
	# matrix representation of graph
	graph = [[0, 10, 15, 20], [10, 0, 35, 25],
			 [15, 35, 0, 30], [20, 25, 30, 0]]
	s = 0
	print("graph: ", graph)
	print("solution:", travellingSalesmanProblem(graph, s))
	# should result 80


def main():
	# Definição de argumentos
	parser = argparse.ArgumentParser(description='Naive TPS')
	help_msg = "arquivo de saída.  Padrão:{}".format(DEFAULT_OUT)
	parser.add_argument("--out", "-o", help=help_msg, default=DEFAULT_OUT, type=str)

	help_msg = "semente aleatória. Padrão:{}".format(DEFAULT_SEED)
	parser.add_argument("--seed", "-s", help=help_msg, default=DEFAULT_SEED, type=int)

	help_msg = "n máximo.          Padrão:{}".format(DEFAULT_N_STOP)
	parser.add_argument("--nstop", "-n", help=help_msg, default=DEFAULT_N_STOP, type=int)

	help_msg = "n mínimo.          Padrão:{}".format(DEFAULT_N_START)
	parser.add_argument("--nstart", "-a", help=help_msg, default=DEFAULT_N_START, type=int)

	help_msg = "n passo.           Padrão:{}".format(DEFAULT_N_STEP)
	parser.add_argument("--nstep", "-e", help=help_msg, default=DEFAULT_N_STEP, type=int)

	help_msg = "tentativas.        Padrão:{}".format(DEFAULT_N_STEP)
	parser.add_argument("--trials", "-t", help=help_msg, default=DEFAULT_TRIALS, type=int)

	# Lê argumentos from da linha de comando
	args = parser.parse_args()

	#teste()
	trials = args.trials
	f = open(args.out, "w")
	f.write("#Naive solution for the Travelling Salesman Problem (TSP)\n")
	f.write("#n time_s_avg time_s_std result_avg result_std (for {} trials)\n".format(trials))
	m = 100
	np.random.seed(args.seed)

	for n in range(args.nstart, args.nstop+1, args.nstep): #range(1, 100):
		resultados = [0 for i in range(trials)]
		tempos = [0 for i in range(trials)]
		for trial in range(trials):
			print("\n-------")
			print("n: {} trial: {}".format(n, trial+1))
			graph = planar_graph_generator(n, m)
			tempo_inicio = timeit.default_timer()
			resultados[trial] = travellingSalesmanProblem(graph, 0)
			tempo_fim = timeit.default_timer()
			tempos[trial] = tempo_fim - tempo_inicio
			print("Saída: {}".format(resultados[trial]))
			print('Tempo: {} s'.format(tempos[trial]))
			print("")

		tempos_avg = np.average(tempos)  # calcula média
		tempos_std = np.std(a=tempos, ddof=False)  # ddof=calcula desvio padrao de uma amostra?

		resultados_avg = np.average(resultados)  # calcula média
		resultados_std = np.std(a=resultados, ddof=False)  # ddof=calcula desvio padrao de uma amostra?

		f.write("{} {} {} {} {}\n".format(n, tempos_avg, tempos_std, resultados_avg, resultados_std))
	f.close()


if __name__ == '__main__':
	sys.exit(main())
