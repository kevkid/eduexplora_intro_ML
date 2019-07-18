# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 11:31:18 2018

@author: kevin
"""

#Lecture 7
### HC clustering########
from sklearn.datasets.samples_generator import make_blobs
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
X, y = make_blobs(n_samples=100, centers=3, n_features=2, random_state=92)#best seed to demonstrate
#before assigning cluster

# Using the dendrogram to find the optimal number of clusters

import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
plt.title('Dendrogram')
plt.xlabel('Points')
plt.ylabel('Euclidean distances')
plt.show()

plt.scatter(X[:,0], X[:,1])
# Fitting Hierarchical Clustering to the dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 3, affinity = 'euclidean', linkage = 'ward')
y_hat = hc.fit_predict(X)
#after assigning clusters
plt.scatter(X[:,0], X[:,1], c = y_hat, cmap = ListedColormap(('red', 'green', 'blue')))


# Okay how about this, just try and figure out the number of clusters:
X, y = make_blobs(n_samples=100, centers=int('11', 2), n_features=2, random_state=800)#best seed to demonstrate
# Using the dendrogram to find the optimal number of clusters
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
plt.title('Dendrogram')
plt.xlabel('Points')
plt.ylabel('Euclidean distances')
plt.show()

plt.scatter(X[:,0], X[:,1])
# Fitting Hierarchical Clustering to the dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = ENTER THE NUMBER OF CLUSTERS HERE!!!!!!, affinity = 'euclidean', linkage = 'ward')
y_hat = hc.fit_predict(X)
#after assigning clusters
plt.scatter(X[:,0], X[:,1], c = y_hat, cmap = ListedColormap(('TYPE', 'THE', 'DIFFERENT', 'COLORS', 'HERE', 'Depending', 'ON', 'Number of clusters')))

#genetic algorithm
import numpy as np
try:
    from randomstate.prng.pcg64 import RandomState
except ImportError:
    print ("""Importing randomstate failed. To fix, try:
    sudo pip install randomstate OR conda install -c dhirschfeld randomstate""")
    import sys
    sys.exit()

gene_bases = [base for base in ' ABCDEFGHIJKLMNOPQRSTUVWXYZ!?@#$%^&*()']

random_seed = 3

size_of_generation = 1000
prngs = [RandomState(random_seed, i) for i in range(size_of_generation)]

def mutate(gene, prng, mutation_rate=0.05):
	copy = ''
	for base in gene:
		if prng.uniform() < mutation_rate:
			copy += prng.choice(gene_bases)
		else:
			copy += base
	return copy

def fitness(gene, reference='I LOVE MACHINE LEARNING!'):
	return sum([1 for base, ref_base in zip(gene, reference) if base == ref_base])

def new_population(parent, mutation_rate=0.05):
	return [mutate(parent, prng, mutation_rate=mutation_rate) for prng in prngs]

def best_in_population(population):
	"""return the fittest individual in the population"""
	return population[np.argmax([fitness(individual) for individual in population])]

def get_next_parent(parent, mutation_rate=0.05):
	"""evolve a new population from the parent, and find the new fittest individual"""
	return best_in_population(new_population(parent, mutation_rate=mutation_rate))

def weasel_program(mutation_rate=0.05, initial='                        '):#must be same size as text we want
	generation = 0
	score = fitness(initial)
	parent = initial
	while score < len(parent):
		print ('%3d  %s  (%d)' % (generation, parent, score))
		parent = get_next_parent(parent)
		generation += 1
		score = fitness(parent)
	print ('%3d  %s  (%d)' % (generation, parent, score))

if __name__ == '__main__':
	import time
	start = time.time()
	weasel_program()
	print ('evolution time:', time.time() - start)
