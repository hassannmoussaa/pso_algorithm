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
    self.particles_pos = np.random.uniform(size=(n_particles, self.particle_dim) ) \
                        * self.init_pos
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

  def update_velocity(self, x, v, p_best, g_best  , clusterGBEST, lbest , i , c0=0.5, c1=1.5, w=0.75):
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
    if clusterGBEST != None :
        new_v = ineteria*v + c0 * r * (p_best - x) + c1 * r * (lbest - x)
    else :
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
      if clusterGBEST != None :
          lbest = getPBest(cluster_particles[clusterGBEST] , self.particles_pos , self.particle_dim)
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

def _safe_accumulator_op(op, x, *args, **kwargs):
    """
    This function provides numpy accumulator functions with a float64 dtype
    when used on a floating point input. This prevents accumulator overflow on
    smaller floating point dtypes.

    Parameters
    ----------
    op : function
        A numpy accumulator function such as np.mean or np.sum
    x : numpy array
        A numpy array to apply the accumulator function
    *args : positional arguments
        Positional arguments passed to the accumulator function after the
        input x
    **kwargs : keyword arguments
        Keyword arguments passed to the accumulator function

    Returns
    -------
    result : The output of the accumulator function passed to this function
    """
    if np.issubdtype(x.dtype, np.floating) and x.dtype.itemsize < 8:
        result = op(x, *args, **kwargs, dtype=np.float64)
    else:
        result = op(x, *args, **kwargs)
    return result


def sphere(x):
  """
    In 3D: f(x,y,z) = x² + y² + z²
  """
  value  = np.sum(np.square(x))
  return value

def JongF4(x):
    return np.sum(np.power(x , 4))

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

def Rosenbrock(chromosome):
	"""F8 Rosenbrock's saddle
	multimodal, asymmetric, inseparable"""
	fitness = 0
	for i in range(len(chromosome)-1):
		fitness += 100*((chromosome[i]**2)-chromosome[i+1])**2+\
		(1-chromosome[i])**2
	return fitness

def Alpine(x):
    sum = 0
    for i in range(0 , dimension_nb):
        sum+= abs(x[i] * sin(x[i]) + 0.1*x[i])
    return sum

def Rastrigin(chromosome):
	"""F5 Rastrigin's function
	multimodal, symmetric, separable"""
	fitness = 10*len(chromosome)
	for i in range(len(chromosome)):
		fitness += chromosome[i]**2 - (10*cos(2*pi*chromosome[i]))
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


def Easom(x):
  """
    In 3D: f(x,y,z) = x² + y² + z²
  """
  return bf.Easom().get_func_val(x)

def Schwefel(chromosome):
	"""F7 Schwefel's function
	multimodal, asymmetric, separable"""
	alpha = 418.982887
	fitness = 0
	for i in range(len(chromosome)):
		fitness -= chromosome[i]*sin(sqrt(fabs(chromosome[i])))
	return float(fitness) + alpha*len(chromosome)


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

def ShiftedRosenbroc(x):
    f_bias = 390
    F = 0
    dim = len(x)
    z = np.empty(dim)
    for i in range(dim - 1):
        z[i] = x[i] - PSO_s.g_best[i] + 1
    for i in range (dim - 2):
        F += 100 * ((z[i]**2 - z[i + 1])**2) + (z[i] - 1)**2
    res = F + f_bias
    return res

# Define the Shifted Rastrigin's function with the previous parameters
def ShiftedRastrigin(x):
    f_bias =  -330
    F = 0
    dim = len(x)
    for i in range(dim - 1):
        z = x[i] - PSO_s.g_best[i]
        F += (z ** 2) - (10 * cos(2 * pi * z)) + 10
    res = F + f_bias
    return res


def ShiftedGriwank(x):
    f_bias =  -180
    F1 = 0
    F2 = 1
    dim = len(x)
    for i in range(dim - 1):
        z = x[i] - PSO_s.g_best[i]
        F1 += z ** 2 /4000
        F2 += cos(z / sqrt(i + 1))
    F = F1 - F2 + 1
    res = F + f_bias
    return res


# Define the Shifted Ackley's function with the previous parameters
def ShiftedAckley(x):
    f_bias = -140
    dim = len(x)
    Sum1 = 0
    Sum2 = 0
    for i in range(dim - 1):
        z = x[i] - PSO_s.g_best[i]
        Sum1 += z ** 2
        Sum2 += cos(2 * pi * z)
    Sum = -20 * exp(-0.2 * sqrt(Sum1 / dim)) - exp(Sum2 / dim) + 20 + e
    res = Sum + f_bias
    return res


def ShiftedSphere(x):
    f_bias = -450
    dim = len(x)
    F = 0
    for i in range(dim - 1):
        z = x[i] - PSO_s.g_best[i]
        F += z**2
    res = F + f_bias
    return res

def ShiftedShwefel(x):
    f_bias = -450
    dim = len(x)
    F = abs(x[0] - PSO_s.g_best[0])
    for i in range(1, dim - 1):
        z = x[i] - PSO_s.g_best[i]
        F = max(F, abs(z))
    res = F + f_bias
    return res

def Clustering(X):
  from sklearn.cluster import KMeans
  from sklearn.cluster import MeanShift, estimate_bandwidth
  from sklearn.cluster import DBSCAN
  from sklearn.preprocessing import StandardScaler
  from sklearn.decomposition import PCA
  from numpy import unique
  ##################### CLUSTERING ##########################################################
  model = DBSCAN(eps=0.30, min_samples=9)
  model.fit(X)
  y_kmeans = model.fit_predict(X)
  clusters__ = unique(y_kmeans)
  n_cluster = len(clusters__)
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
            for t in range(0 , dimension_nb):
      
                if cluster_particles[i][0][j][t] != part_value[t]:
                    break
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
        if sphere(getPBestOfAParticle(particles , cluster[0][i] ,dimension_nb)) < min_value:
            min_value =   sphere(getPBestOfAParticle(particles , cluster[0][i] ,dimension_nb))
    return min_value
        
if __name__ == '__main__':  
  start_time = time.time()
  dimension_nb = 50
  init_pos = np.ones(shape=( dimension_nb))
  f = open("resultat.txt", "w")
  print(">>>>>>>>>>>>>> FUNCTION SPHERE >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
  PSO_s = PSO(func=sphere, init_pos=init_pos, n_particles=150)
  cluster_particles =  Clustering(PSO_s.particles_pos)
  res_s= PSO_s.optimize(200, cluster_particles)
  print(">>>>"  , res_s[1])
  print(">>>>"  , res_s[0])
  f.write(" #1 Function  Sphere-- > Time Used :  " +  str((time.time() - start_time)) + " Value  : " + str(res_s[1]) + " Position " + ''.join(str(e) for e in res_s[0]) + " \n" )

  print("--- %s seconds ---" % (time.time() - start_time))
  print(">>>>>>>>>>>>>> END FUNCTION SPHERE >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
  
  
  print(">>>>>>>>>>>>>> De Jong's f4 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
  PSO_s = PSO(func=JongF4, init_pos=init_pos, n_particles=50)
  cluster_particles =  Clustering(PSO_s.particles_pos)
  res_s= PSO_s.optimize(300, cluster_particles)
  print(">>>>"  , res_s[1])
  print(">>>>"  , res_s[0])
  f.write(" #2 Function  De Jong's f4  -- > Time Used :  " + str((time.time() - start_time)) + " Value  : " + str(res_s[1]) + " Position " + ''.join(str(e) for e in res_s[0]) + " \n" )

  print("--- %s seconds ---" % (time.time() - start_time))
  print(">>>>>>>>>>>>>> END De Jong's f4 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
  
  print(">>>>>>>>>>>>>> Griwank >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
  PSO_s = PSO(func=Griwank, init_pos=init_pos, n_particles=50)
  cluster_particles =  Clustering(PSO_s.particles_pos)
  res_s= PSO_s.optimize(300, cluster_particles)
  print(">>>>"  , res_s[1])
  print(">>>>"  , res_s[0])
  f.write(" #3 Function  Griwank -- > Time Used :  " + str((time.time() - start_time)) + " Value  : " + str(res_s[1]) + " Position " + ''.join(str(e) for e in res_s[0]) + " \n" )

  print("--- %s seconds ---" % (time.time() - start_time))
  print(">>>>>>>>>>>>>> END Griwank >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
  
  print(">>>>>>>>>>>>>> Rosenbrock >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
  PSO_s = PSO(func=Rosenbrock, init_pos=init_pos, n_particles=50)
  cluster_particles =  Clustering(PSO_s.particles_pos)
  res_s= PSO_s.optimize(300, cluster_particles)
  print(">>>>"  , res_s[1])
  print(">>>>"  , res_s[0])
  f.write(" #4 Function  Rosenbrock -- > Time Used :  " + str((time.time() - start_time)) + " Value  : " + str(res_s[1]) + " Position " + ''.join(str(e) for e in res_s[0]) + " \n" )

  print("--- %s seconds ---" % (time.time() - start_time))
  print(">>>>>>>>>>>>>> END Rosenbrock >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
  
  
  print(">>>>>>>>>>>>>> Alpine function, >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
  PSO_s = PSO(func=Alpine, init_pos=init_pos, n_particles=50)
  cluster_particles =  Clustering(PSO_s.particles_pos)
  res_s= PSO_s.optimize(300, cluster_particles)
  print(">>>>"  , res_s[1])
  print(">>>>"  , res_s[0])
  f.write(" #5 Function  Alpine -- > Time Used :  " + str((time.time() - start_time)) + " Value  : " + str(res_s[1]) + " Position " + ''.join(str(e) for e in res_s[0]) + " \n" )

  print("--- %s seconds ---" % (time.time() - start_time))
  print(">>>>>>>>>>>>>> END Alpine >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
  
  print(">>>>>>>>>>>>>> Rastrigin function, >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
  PSO_s = PSO(func=Rastrigin, init_pos=init_pos, n_particles=50)
  cluster_particles =  Clustering(PSO_s.particles_pos)
  res_s= PSO_s.optimize(300, cluster_particles)
  print(">>>>"  , res_s[1])
  print(">>>>"  , res_s[0])
  f.write(" #6 Function  Rastrigin -- > Time Used :  " + str((time.time() - start_time)) + " Value  : " + str(res_s[1]) + " Position " + ''.join(str(e) for e in res_s[0]) + " \n" )

  print("--- %s seconds ---" % (time.time() - start_time))
  print(">>>>>>>>>>>>>> END Rastrigin >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
  
  
  print(">>>>>>>>>>>>>> Ackley function, >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
  PSO_s = PSO(func=Ackley, init_pos=init_pos, n_particles=50)
  cluster_particles =  Clustering(PSO_s.particles_pos)
  res_s= PSO_s.optimize(300, cluster_particles)
  print(">>>>"  , res_s[1])
  print(">>>>"  , res_s[0])
  f.write(" #7 Function  Ackley -- > Time Used :  " + str((time.time() - start_time)) + " Value  : " + str(res_s[1]) + " Position " + ''.join(str(e) for e in res_s[0]) + " \n" )

  print("--- %s seconds ---" % (time.time() - start_time))
  print(">>>>>>>>>>>>>> END Ackley >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
  
  
  print(">>>>>>>>>>>>>> Easom function, >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
  PSO_s = PSO(func=Easom, init_pos=[1 , 1], n_particles=150)
  cluster_particles =  Clustering(PSO_s.particles_pos)
  res_s= PSO_s.optimize(50, cluster_particles)
  print(">>>>"  , res_s[1])
  print(">>>>"  , res_s[0])
  f.write(" #8 Function  Easom -- > Time Used :  " + str((time.time() - start_time)) + " Value  : " + str(res_s[1]) + " Position " + ''.join(str(e) for e in res_s[0]) + " \n" )

  print("--- %s seconds ---" % (time.time() - start_time))
  print(">>>>>>>>>>>>>> END Easom >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
  
  print(">>>>>>>>>>>>>> Schwefel function, >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
  PSO_s = PSO(func=Schwefel, init_pos=init_pos, n_particles=50)
  cluster_particles =  Clustering(PSO_s.particles_pos)
  res_s= PSO_s.optimize(50, cluster_particles)
  print(">>>>"  , res_s[1])
  print(">>>>"  , res_s[0])
  f.write(" #9 Function  Schwefel -- > Time Used :  " + str((time.time() - start_time)) + " Value  : " + str(res_s[1]) + " Position " + ''.join(str(e) for e in res_s[0]) + " \n" )

  print("--- %s seconds ---" % (time.time() - start_time))
  print(">>>>>>>>>>>>>> END Schwefel >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
  
  
  print(">>>>>>>>>>>>>> Exponotial function, >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
  PSO_s = PSO(func=ExponotialFunction, init_pos=init_pos, n_particles=50)
  cluster_particles =  Clustering(PSO_s.particles_pos)
  res_s= PSO_s.optimize(50, cluster_particles)
  print(">>>>"  , res_s[1])
  print(">>>>"  , res_s[0])
  f.write(" #10 Function  Exponotial -- > Time Used :  " + str((time.time() - start_time)) + " Value  : " + str(res_s[1]) + " Position " + ''.join(str(e) for e in res_s[0]) + " \n" )
  print("--- %s seconds ---" % (time.time() - start_time))
  print(">>>>>>>>>>>>>> END ExponotialFunction >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
  
  
  print(">>>>>>>>>>>>>>  Shifted Rosenbrock's  function, >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
  PSO_s = PSO(func=ShiftedRosenbroc, init_pos=init_pos, n_particles=50)
  cluster_particles =  Clustering(PSO_s.particles_pos)
  res_s= PSO_s.optimize(50, cluster_particles)
  print(">>>>"  , res_s[1])
  print(">>>>"  , res_s[0])
  f.write(" #11  Function Shifted Rosenbrock -- > Time Used :  " + str((time.time() - start_time)) + " Value  : " + str(res_s[1]) + " Position " + ''.join(str(e) for e in res_s[0]) + " \n" )

  print("--- %s seconds ---" % (time.time() - start_time))
  print(">>>>>>>>>>>>>> END  Shifted Rosenbrock >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
  
  print(">>>>>>>>>>>>>>  Shifted Rastring's  function, >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
  PSO_s = PSO(func=ShiftedRastrigin, init_pos=init_pos, n_particles=50)
  cluster_particles =  Clustering(PSO_s.particles_pos)
  res_s= PSO_s.optimize(50, cluster_particles)
  print(">>>>"  , res_s[1])
  print(">>>>"  , res_s[0])
  f.write(" #12 Function Shifted Rastring -- > Time Used :  " + str((time.time() - start_time)) + " Value  : " + str(res_s[1]) + " Position " + ''.join(str(e) for e in res_s[0]) + " \n" )

  print("--- %s seconds ---" % (time.time() - start_time))
  print(">>>>>>>>>>>>>> END  Rastring Rosenbrock >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
  
  
  print(">>>>>>>>>>>>>>   Shifted Griwank's  function, >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
  PSO_s = PSO(func=ShiftedGriwank, init_pos=init_pos, n_particles=50)
  cluster_particles =  Clustering(PSO_s.particles_pos)
  res_s= PSO_s.optimize(200, cluster_particles)
  print(">>>>"  , res_s[1])
  print(">>>>"  , res_s[0])
  f.write(" #13 Function Shifted Griwank -- > Time Used :  " + str((time.time() - start_time)) + " Value  : " + str(res_s[1]) + " Position " + ''.join(str(e) for e in res_s[0]) + " \n" )

  print("--- %s seconds ---" % (time.time() - start_time))
  print(">>>>>>>>>>>>>> END  Shifted Griwank Rosenbrock >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
  
  print(">>>>>>>>>>>>>>   Shifted Ackley's  function, >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
  PSO_s = PSO(func=ShiftedAckley, init_pos=init_pos, n_particles=50)
  cluster_particles =  Clustering(PSO_s.particles_pos)
  res_s= PSO_s.optimize(200, cluster_particles)
  print(">>>>"  , res_s[1])
  print(">>>>"  , res_s[0])
  f.write(" #14 Function Shifted Ackley -- > Time Used :  " + str((time.time() - start_time)) + " Value  : " + str(res_s[1]) + " Position " + ''.join(str(e) for e in res_s[0]) + " \n" )

  print("--- %s seconds ---" % (time.time() - start_time))
  print(">>>>>>>>>>>>>> END  Shifted Ackley Rosenbrock >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
  
  
  print(">>>>>>>>>>>>>>   Shifted Sphere  function, >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
  PSO_s = PSO(func=ShiftedSphere, init_pos=init_pos, n_particles=50)
  cluster_particles =  Clustering(PSO_s.particles_pos)
  res_s= PSO_s.optimize(200, cluster_particles)
  print(">>>>"  , res_s[1])
  print(">>>>"  , res_s[0])
  f.write(" #15 Function Shifted Sphere -- > Time Used :  " + str((time.time() - start_time)) + " Value  : " + str(res_s[1]) + " Position " + ''.join(str(e) for e in res_s[0]) + " \n" )

  print("--- %s seconds ---" % (time.time() - start_time))
  print(">>>>>>>>>>>>>> END  Shifted Sphere Rosenbrock >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
  
  
  print(">>>>>>>>>>>>>>   Shifted Shwefel  function, >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
  PSO_s = PSO(func=ShiftedShwefel, init_pos=init_pos, n_particles=50)
  cluster_particles =  Clustering(PSO_s.particles_pos)
  res_s= PSO_s.optimize(200, cluster_particles)
  print(">>>>"  , res_s[1])
  print(">>>>"  , res_s[0])
  f.write(" #16 Function Shifted Shwefel -- > Time Used :  " + str((time.time() - start_time)) + " Value  : " + str(res_s[1]) + " Position " + ''.join(str(e) for e in res_s[0]) + " \n" )

  print("--- %s seconds ---" % (time.time() - start_time))
  print(">>>>>>>>>>>>>> END  Shifted Shwefel Rosenbrock >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
  
  f.close()
  
  
  
  
  
  
  
  
  
  
  