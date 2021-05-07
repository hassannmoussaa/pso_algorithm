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
import random as rd

wMin = 0.2

phiMin = 2
phiMax = 11
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import silhouette_score
class MDCLUSTER_PSO(object):
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
    self.phi = 2
  def sortParticles(self):
      self.fitnesses = []
      for i in range(0 , self.n_particles):
          item = {"index" : i , "fitness" : self.func(self.particles_pos[i]) }
          self.fitnesses.append(item )
      self.fitnesses.sort(key=lambda fit: fit.get('fitness') ,  reverse = True)
  def cluster(self):
      last_phi = self.phi
      self.clusters = []
      delimeter = 0
      for i in range(delimeter , self.n_particles):
              if delimeter == self.n_particles :
                  break
              cluster = self.fitnesses[delimeter : delimeter+self.phi]
              delimeter +=  self.phi 
              self.clusters.append(cluster)     
  def ConnectClusters(self):
      for i in range(0 , len(self.clusters) - 1):
          item = {'index' : 'connected' , 'fitness' : self.getLbestOfCluster(i+1) }
          self.clusters[i].append(item)
  def getClusterOfParticle(self , index):
      for i in range(0 , len(self.clusters)):
          for j in range(0 , len(self.clusters[i])):
              if self.clusters[i][j].get('index') == index :
                  return i
  def getLbestOfCluster(self , index):
      l_best = 0
      for i in range(0 , len(self.clusters[index])):
          if self.clusters[index][i].get('fitness') <= l_best :
              l_best = self.clusters[index][i].get('fitness')
      return l_best
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

  def update_velocity(self, x, v, p_best, g_best , lbest , i , c0=1.5, c1=1.5, w=0.75):
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
      self.sortParticles()
      self.phi =  int(phiMin + iteration *((phiMax - phiMin) / float(maxiter)))
      self.cluster()
      #self.ConnectClusters()
      for i in range(self.n_particles):
          x = self.particles_pos[i]
          v = self.velocities[i]
          p_best = self.p_best[i]
          self.velocities[i] = self.update_velocity(x, v, p_best , self.g_best , self.getLbestOfCluster(self.getClusterOfParticle(i)) , i)
          self.particles_pos[i] = self.update_position(x, v)
          # Update the best position for particle i
          if self.func(self.particles_pos[i]) < self.func(p_best):
              self.p_best[i] = self.particles_pos[i]
          # Update the best position overall
          if self.func(self.particles_pos[i]) < self.func(self.g_best):
              
              self.g_best = self.particles_pos[i]
    return self.g_best, self.func(self.g_best)
         
# Example of the sphere function
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
def sphere(x):
  """
    In 3D: f(x,y,z) = x² + y² + z²
  """
  return np.sum(np.square(x))

maxIteration = 200

start_time = time.time()
dimension_nb = 2
init_pos = np.ones(dimension_nb)
PSO_s = MDCLUSTER_PSO(func=sphere, init_pos=init_pos, n_particles=50)
res_s= PSO_s.optimize(maxIteration)
print(">>>>"  , res_s[1])
print(">>>>"  , res_s[0])
print("--- %s seconds ---" % (time.time() - start_time))