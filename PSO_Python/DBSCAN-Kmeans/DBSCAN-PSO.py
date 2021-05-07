# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 19:53:19 2021

@author: hassa
"""
import time
import numpy as np
import matplotlib.pyplot as plt
from math import *
wMax = 0.9 
import sys
import pandas as pd
wMin = 0.2
maxIteration = 10
from scipy import spatial
from sklearn.metrics.pairwise import cosine_distances

from opteval import benchmark_func as bf

import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import silhouette_score
class PSO(object):
  """
    Class implementing PSO algorithm.
  """
  def __init__(self, func, init_pos, n_particles):
    """
      Initialize the key variables.
      Args:
        func (function): the fitness function to optimize.
        init_pos (array-like): the initial position to kick off the
                               optimization process.
        n_particles (int): the number of particles of the swarm.
    """
    self.func = func
    self.n_particles = n_particles
    self.init_pos = np.array(init_pos)
    self.particle_dim = len(init_pos)
    # Initialize particle positions using a uniform distribution
    self.particles_pos = np.random.uniform(-2 , 2 , size=(n_particles, self.particle_dim))
    # Initialize particle velocities using a uniform distribution
    self.velocities = np.random.uniform(size=(n_particles, self.particle_dim))

    # Initialize the best positions
    self.g_best = init_pos
    self.p_best = self.particles_pos

  def update_position(self, x, v):
    """
      Update particle position.
      Args:
        x (array-like): particle current position.
        v (array-like): particle current velocity.
      Returns:
        The updated position (array-like).
    """
    x = np.array(x)
    v = np.array(v)
    new_x = x + v
    return new_x

  def update_velocity(self, x, v, p_best, g_best  , clusterGBEST, lbest , i , c0=1.5, c1=1.5, w=0.75):
    """
      Update particle velocity.
      Args:
        x (array-like): particle current position.
        v (array-like): particle current velocity.
        p_best (array-like): the best position found so far for a particle.
        g_best (array-like): the best position regarding
                             all the particles found so far.
        c0 (float): the cognitive scaling constant.
        c1 (float): the social scaling constant.
        w (float): the inertia weight
      Returns:
        The updated velocity (array-like).
    """
    ineteria = wMax - i *((wMax - wMin) / float(maxIteration))

    x = np.array(x)
    v = np.array(v)
    assert x.shape == v.shape, 'Position and velocity must have same shape'
    # a random number between 0 and 1.
    r = np.random.uniform()
    p_best = np.array(p_best)
    g_best = np.array(g_best)
    new_v=0
    if clusterGBEST != None :
        new_v = ineteria*v + c0 * r * (p_best - x) + c1 * r * (lbest - x)
    else:
        new_v = ineteria*v + c0 * r * (p_best - x) + c1 * r * (g_best - x)
        
    return new_v

  def optimize(self, maxiter , cluster_particles  ):
    """
      Run the PSO optimization process untill the stoping criteria is met.
      Case for minimization. The aim is to minimize the cost function.
      Args:
          maxiter (int): the maximum number of iterations before stopping
                         the optimization.
      Returns:
          The best solution found (array-like).
    """
    for _ in range(maxiter):
      cluster_particles =  Clustering(PSO_s.particles_pos)
      lbest = 0 
      clusterGBEST =  getClusterOfAParticle(cluster_particles ,PSO_s.g_best , self.particle_dim )
     # print(clusterGBEST)
      if clusterGBEST != None :
          lbest = getPBest(cluster_particles[clusterGBEST] , self.particles_pos , self.particle_dim)
      #print(lbest)
      for i in range(self.n_particles):
          x = self.particles_pos[i]
          v = self.velocities[i]
          p_best = self.p_best[i]
          self.velocities[i] = self.update_velocity(x, v, p_best , self.g_best ,  clusterGBEST ,lbest , i)
          self.particles_pos[i] = self.update_position(x, v)
          # Update the best position for particle i
          if self.func(self.particles_pos[i]) < self.func(p_best):
              self.p_best[i] = self.particles_pos[i]
          # Update the best position overall
          if self.func(self.particles_pos[i]) < self.func(self.g_best):
              
              self.g_best = self.particles_pos[i]
    return self.g_best, self.func(self.g_best)
# Example of the sphere function


def cossim(x , y):
    return  1 - spatial.distance.cosine(x, y)

def sphere(x):
  """
    In 3D: f(x,y,z) = x² + y² + z²
  """
  value  = np.sum(np.square(x))
  return value


def Clustering(X):
  from sklearn.cluster import KMeans
  from sklearn.cluster import MeanShift, estimate_bandwidth
  from sklearn.cluster import DBSCAN
  from sklearn.preprocessing import StandardScaler
  from sklearn.decomposition import PCA
  from numpy import unique
  ##################### CLUSTERING ##########################################################
  model = DBSCAN(eps=20, min_samples=2)
  model.fit(X)
  y_kmeans = model.fit_predict(X)
  clusters__ = unique(y_kmeans)
  n_cluster = len(clusters__)
 # print(n_cluster)
  clusters = np.zeros((n_cluster , len(y_kmeans)))
  positions = []
  for i in range(0 , n_cluster):
      for j in range(0 , len(y_kmeans)):
          if y_kmeans[j] == i :
              clusters[i , j] = 1
  full_clusters = []
  for i in range(0 , n_cluster):    
      i_cluster = np.where(clusters[i] == 1)
      full_clusters.append(i_cluster)
  clusters_particles = []
  for i in range(0,n_cluster):
      cluster_particle = []
      for j in range(0 , len(full_clusters[i])):
          cluster_particle.append(X[full_clusters[i][j]])
      clusters_particles.append(cluster_particle)
  return  clusters_particles

def getClusterOfAParticle(cluster_particles , part_value , dimension_nb):
    for i in range(0 , len(cluster_particles)):
        for j in range(0 , len(cluster_particles[i][0])):
                comparison = np.array(cluster_particles[i][0][j]) == np.array(part_value)
                if  comparison.all():
                    return i
            
def getPBestOfAParticle(particles ,  part_value , dimension_nb):
    min_value = inf
    for t in range(0 , len(particles)):
        for k in range(0 ,  dimension_nb):
            if particles[t][k] != part_value[k]:   
                break
            return  PSO_s.p_best[t]
def getPBest(cluster , particles,  dimension_nb ):
    min_value = inf
    for i in range(0 , len(cluster[0])):
        if lenardJonesFunction(getPBestOfAParticle(particles , cluster[0][i] ,dimension_nb)) < min_value:
            min_value =  lenardJonesFunction(getPBestOfAParticle(particles , cluster[0][i] ,dimension_nb))
    return min_value
        
def lenardJonesFunction(x):
    result = 0
    flag = 0 
    D = len(x)
    nb_atomes = int(D/3)
    for i in range(1 , nb_atomes):
        if flag == 0 :
            for j in range(i+1 , nb_atomes+1):
                if flag == 0 :
                    xi = x[(i-1)*3]
                    yi = x[(i-1)*3+1]
                    zi = x[(i-1)*3+2]
                    xj = x[(j-1)*3]
                    yj = x[(j-1)*3+1]
                    zj = x[(j-1)*3+2]
                    d0 = xi-xj
                    d2 = yi-yj 
                    d3 = zi-zj
                    rij = sqrt(abs((d0 * d0) + (d2 * d2) + (d3 * d3)))
                    if rij <= 0.0:
                       flag = 1;
                    else :    
                        result = result + (pow(rij, -12) -pow(rij, -6))
    result= 4*result;
    if flag == 1:
        return inf
    return result
            
def lennardGetRij(x):
    result = 0
    flag = 0 
    D = len(x)
    nb_atomes = int(D/3)
    for i in range(1 , nb_atomes):
            for j in range(i+1 , nb_atomes+1):
                    xi = x[(i-1)*3]
                    yi = x[(i-1)*3+1]
                    zi = x[(i-1)*3+2]
                    xj = x[(j-1)*3]
                    yj = x[(j-1)*3+1]
                    zj = x[(j-1)*3+2]
                    d0 = xi-xj
                    d2 = yi-yj 
                    d3 = zi-zj
                    rij = sqrt((d0 * d0) + (d2 * d2) + (d3 * d3))
    return rij

def Rosenbrock(chromosome):
	"""F8 Rosenbrock's saddle
	multimodal, asymmetric, inseparable"""
	fitness = 0
	for i in range(len(chromosome)-1):
		fitness += 100*((chromosome[i]**2)-chromosome[i+1])**2+\
		(1-chromosome[i])**2
	return fitness
def Ackley(chromosome):
	""""""
	firstSum = 0.0
	secondSum = 0.0
	for c in chromosome:
		firstSum += c**2.0
		secondSum += cos(2.0*pi*c)
	n = float(len(chromosome))
	return -20.0*exp(-0.2*sqrt(firstSum/n)) - exp(secondSum/n) + 20 + e
if __name__ == '__main__': 
  values = 0
  for itera in range(0 ,10):
      start_time = time.time()
      dimension_nb = 27
      init_pos = np.ones(shape=(dimension_nb))
      PSO_s = PSO(func=lenardJonesFunction, init_pos=init_pos, n_particles=150)
      cluster_particles =  Clustering(PSO_s.particles_pos)
      res_s= PSO_s.optimize(150, cluster_particles)
      print("ITERATIOM "  , itera+1)
      print(">>>>"  , res_s[1])
      print(">>>>"  , res_s[0])
      values +=  res_s[1]
      t = lennardGetRij(res_s[0])
      print("--- %s seconds ---" % (time.time() - start_time))
  mean_best = values/10 