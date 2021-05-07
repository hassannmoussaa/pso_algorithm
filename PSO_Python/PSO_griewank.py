# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 19:53:19 2021

@author: hassa
"""
import numpy as np
import time
from math import *

import matplotlib.pyplot as plt
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
    self.particles_pos =  np.random.uniform(size=(n_particles, self.particle_dim)) \
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

  def update_velocity(self, x, v, p_best, g_best , c0=2, c1 = 2, w=0.55):
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
   
    x = np.array(x)
    v = np.array(v)
    assert x.shape == v.shape, 'Position and velocity must have same shape'
    # a random number between 0 and 1.
    r = np.random.uniform()
    p_best = np.array(p_best)
    g_best = np.array(g_best)

    new_v = w*v + c0 * r * (p_best - x) + c1 * r * (g_best - x)
    return new_v

  def optimize(self, maxiter=500):
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
          self.velocities[i] = self.update_velocity(x, v, p_best, self.g_best )
          self.particles_pos[i] = self.update_position(x, v)
          # Update the best position for particle i
          if self.func(self.particles_pos[i]) < self.func(p_best):
              self.p_best[i] = self.particles_pos[i]
          # Update the best position overall
          if self.func(self.particles_pos[i]) < self.func(self.g_best):
              self.g_best = self.particles_pos[i]
    return self.g_best, self.func(self.g_best)
  
# Example of the sphere function
# Example of the sphere function
# Example of the sphere function

def fitnessFunc(chromosome):
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
  failed = 0
  for i in range(0 , 100) :
      
      init_pos = [1,1 ]
      PSO_s = PSO(func=fitnessFunc, init_pos=init_pos, n_particles=50)
      res_s = PSO_s.optimize()
      if res_s[1] > 0.3:
          failed += 1
      print(">>>>"  , res_s[1])
      print(">>>>"  , res_s[0])
      print("--- %s seconds ---" % (time.time() - start_time))

  print("FAILUR" , failed)
  
  
         
        
  #res_s = PSO_s.optimize()
  #print("Sphere function")
  #print(f'x = {res_s[0]}') # x = [-0.00025538 -0.00137996  0.00248555]
  #print(f'f = {res_s[1]}') # f = 8.14748063004205e-06