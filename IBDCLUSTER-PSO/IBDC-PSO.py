# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 19:53:19 2021

@author: hassa
"""
import random

import time
import numpy as np
import matplotlib.pyplot as plt
from math import *
wMax = 0.9 
wMin = 0.4
from opteval import benchmark_func as bf

maxIteration = 500
phiMax = 10
phiMin = 2
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
    self.particles_pos = np.random.uniform(size=(n_particles, self.particle_dim) ) \
                        * self.init_pos
    # Initialize particle velocities using a uniform distribution
    self.velocities = np.random.uniform(size=(n_particles, self.particle_dim))
    # Initialize the best positions
    self.g_best = init_pos
    self.p_best = self.particles_pos
    self.clusters = []
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

  def update_velocity(self, x, v, p_best, g_best , lbest  , i , c0=1.5, c1=1.5 , w=0.75):
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
    new_v = ineteria*v + c0 * r * (p_best - x) + c1 * r * (lbest - x)
    return new_v

  def optimize(self, maxiter):
    """
      Run the PSO optimization process untill the stoping criteria is met.
      Case for minimization. The aim is to minimize the cost function.
      Args:
          maxiter (int): the maximum number of iterations before stopping
                         the optimization.
      Returns:
          The best solution found (array-like).
    """
    for iteration in range(maxiter):
      phi = int(phiMax - iteration *((phiMax - phiMin) / float(maxiter)))
      kernels = GetKernels(phi , PSO_s.particles_pos )
      self.clusters = []
      self.clusters = clusters = GetClusters(kernels)
      ineteria = wMax -  iteration *((wMax - wMin) / float(maxIteration))
      self.clusters = clusters = cluster(clusters , self.particles_pos , ineteria , self.p_best , self.g_best , 0.5 , 1.5 ,  self.velocities)
      self.clusters = ConnectClusters( self.clusters)
      
      for i in range(self.n_particles):
          clusterofparticle = GetClusterOfAPartile(self.clusters , self.particles_pos[i])
          lbest = getLBestOfCluster(self.clusters[clusterofparticle])
          
          x = self.particles_pos[i]
          v = self.velocities[i]
          p_best = self.p_best[i]
          self.velocities[i] = self.update_velocity(x, v, p_best , self.g_best , lbest  , iteration)
          self.particles_pos[i] = self.update_position(x, v)
          # Update the best position for particle i
          if self.func(self.particles_pos[i]) < self.func(p_best):
              self.p_best[i] = self.particles_pos[i]
          # Update the best position overall
          if self.func(self.particles_pos[i]) < self.func(self.g_best):
              self.g_best = self.particles_pos[i]
    return self.g_best, self.func(self.g_best)
# Example of the sphere function
def sphere(x):
  """
    In 3D: f(x,y,z) = x² + y² + z²
  """
  return np.sum(np.square(x))

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
                    rij = sqrt((d0 * d0) + (d2 * d2) + (d3 * d3))
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
    
def ExponotialFunction(chromosome):
	"""F3 Exponential function
	unimodal, asymmetric, separable"""
	alpha = 0
	for i in range(1, len(chromosome)+1):
		alpha += e**(-5.12*i)
	alpha = -alpha
	fitness = 0
	for i in range(len(chromosome)):
		fitness += e**((i+1)*chromosome[i])
	return fitness + alpha

    
def GetKernels(phi,particles):
    kernels_nb = int(len(particles)/ phi)
    return random.sample(list(particles), kernels_nb)

def GetClusters(kernels):
    clusters = []
    for i in range(0 , len(kernels)):
        cluster = []
        cluster.append(kernels[i])
        clusters.append(cluster)
    return clusters
    

def getLBestOfCluster(cluster):
    lbest = 10 
    for i in  range(0 , len(cluster)):
        if sphere(cluster[i]) < lbest:
            lbest = sphere(cluster[i])
    return lbest
            
def getProbaList(particle, clusters , ineteria ,p_best ,  gbest , c0 ,  c1 , v ):
    probas = []
    for c in range(0 , len(clusters)):
        r = np.random.uniform()
        lbest = getLBestOfCluster(clusters[c])
        new_x = particle + ineteria*v + c0 * r * (p_best - particle) + c1 * r * (lbest - particle)
        impact = 0
        for d in range(0 , len(new_x)):
            impact += ((new_x[d] - particle[d])/gbest[d])
        try:
            proba =  np.random.uniform(0 , 1) *  1/((abs(impact)))
        except OverflowError as err:
            proba = 1              
        probas.append(proba)
    return probas

def getMinProba(probas):
    maxProba = 0
    minIndex = 1
    for i in range(0 , len(probas)):
        if probas[i] >=  maxProba :
            maxProba = probas[i]
            minIndex = i 
    return minIndex
            
def cluster(clusters , particles , ineteria , p_best , gbest , c0 , c1 , v ):
    for i in range(0 , len(particles)):
        probas = getProbaList(particles[i] , clusters , ineteria ,  p_best[i] , gbest , c0 ,c1 , v[i]    )
        clusterToAdd = getMinProba(probas)
        for j in range(0 , len(clusters)):
            if j == clusterToAdd :
                clusters[j].append(particles[i])
    return clusters

def ConnectClusters(clusters):
    for i in range(0 , len(clusters)):
        if i == 0:
            selected = np.array(random.sample(list(clusters[i+1]), 1))
            clusters[i].append(selected)
        if i >0 and i < len(clusters) - 1 :
            selected_1 = random.sample(list(clusters[i+1]), 1)
            selected_2 = random.sample(list(clusters[i-1]), 1)
            clusters[i].append(selected_1)
            clusters[i].append(selected_2)
        if i == len(clusters) - 1: 
            selected =  np.array(random.sample(list(clusters[i-1]), 1))
            clusters[i].append(selected)
    return clusters

def GetClusterOfAPartile(clusters , particle):
    for i in range(0 , len(clusters)):
        for j in range(0 , len(clusters[i])):
            for d in range(0 , len(particle)):
                if clusters[i][j][d] != particle[d]:
                    break
            return i
    

def Griwank(chromosome):
	"""F6 Griewank's function
	multimodal, symmetric, inseparable"""
	part1 = 0
	for i in range(len(chromosome)):
		part1 += chromosome[i]**2
		part2 = 1
	for i in range(len(chromosome)):
		part2 *= cos(float(chromosome[i]) / sqrt(i+1))
	return 1 + (float(part1)/4000.0) - float(part2)
if __name__ == '__main__':
  start_time = time.time()
  dimension_nb = 6
  init_pos = np.ones(dimension_nb)
  PSO_s = PSO(func=Griwank, init_pos=init_pos, n_particles=50)
  res_s= PSO_s.optimize(maxIteration)
  print(">>>>"  , res_s[1])
  print(">>>>"  , res_s[0])
  print("--- %s seconds ---" % (time.time() - start_time))
 

  