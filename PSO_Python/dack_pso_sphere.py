# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 19:53:19 2021

@author: hassa
"""
import time
import numpy as np
import matplotlib.pyplot as plt
wMax = 0.9 
wMin = 0.2
maxIteration = 10
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
    self.particles_pos = np.random.uniform(size=(n_particles, self.particle_dim)) \
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

  def update_velocity(self, x, v, p_best, g_best , i , c0=0.5, c1=1.5, w=0.75):
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

    new_v = ineteria*v + c0 * r * (p_best - x) + c1 * r * (g_best - x)
    return new_v

  def optimize(self, maxiter=2000):
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
      for i in range(self.n_particles):
          x = self.particles_pos[i]
          v = self.velocities[i]
          p_best = self.p_best[i]
          self.velocities[i] = self.update_velocity(x, v, p_best, self.g_best , i)
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

if __name__ == '__main__':
  start_time = time.time()

  init_pos = [1,1,1,1,1,1,1,1,1,1,1,11,1]
  PSO_s = PSO(func=sphere, init_pos=init_pos, n_particles=50)
  ######################## ELBOWS METHOD FOR GETTING BEST CLUSTERING ##########################
  X = PSO_s.particles_pos
  from sklearn.cluster import KMeans
  from sklearn.cluster import MeanShift, estimate_bandwidth

  distortions = []
 
  # The bandwidth can be automatically estimated
  bandwidth = estimate_bandwidth(X, quantile=.1,
                                   n_samples=500)
  ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
  ms.fit(X)
  labels = ms.labels_
  cluster_centers = ms.cluster_centers_
    
  n_clusters_ = labels.max()+1
  print("BEST" ,n_clusters_ )
  ##################### CLUSTERING ##########################################################
  n_cluster = n_clusters_
  kmeans = KMeans(n_clusters=n_cluster)
  kmeans.fit(X)
  y_kmeans = kmeans.predict(X)
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
  for i in range(0 ,n_cluster):
      particule_cluster = clusters_particles[i][0]
      choice_array = []
      PSO_s = PSO(func=sphere, init_pos=init_pos, n_particles = len(clusters_particles[i][0]))
   
      PSO_s.particles_pos = clusters_particles[i][0]
      res_s = PSO_s.optimize()
     
      print(res_s[0] , res_s[1] )
      positions.append(res_s[0])
  
  PSO_s2 = PSO(func=sphere, init_pos=init_pos, n_particles = 50)
  choice_array_2 = []
  choiced_array_2 = []
  for t in range(0 , len(positions)):
      choice_array_2.append(t)
  choiced_array_2 = np.random.choice(choice_array_2 , 50)
  final_particules_2 = []
  for k in choiced_array_2:
     final_particules_2.append(positions[k])
  PSO_s2.particles_pos = final_particules_2
  res_s = PSO_s2.optimize()
  print(">>>>"  , res_s[1])
  print(">>>>"  , res_s[0])
  print("--- %s seconds ---" % (time.time() - start_time))
  #res_s = PSO_s.optimize()
  #print("Sphere function")
  #print(f'x = {res_s[0]}') # x = [-0.00025538 -0.00137996  0.00248555]
  #print(f'f = {res_s[1]}') # f = 8.14748063004205e-06
  
  
