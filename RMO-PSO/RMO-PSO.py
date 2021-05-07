import numpy as np
from math import *
nb_generation = 10
c1 =0.7
c2 = 0.8
import time
def sphere(x):
  """
    In 3D: f(x,y,z) = x² + y² + z²
  """
  return np.sum(np.square(x))

def Ackley(chromosome):
	""""""
	firstSum = 0.0
	secondSum = 0.0
	for c in chromosome:
		firstSum += c**2.0
		secondSum += cos(2.0*pi*c)
	n = float(len(chromosome))
	return -20.0*exp(-0.2*sqrt(firstSum/n)) - exp(secondSum/n) + 20 + e
def Rastrigin(chromosome):
	"""F5 Rastrigin's function
	multimodal, symmetric, separable"""
	fitness = 10*len(chromosome)
	for i in range(len(chromosome)):
		fitness += chromosome[i]**2 - (10*cos(2*pi*chromosome[i]))
	return fitness

fun = Ackley
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
            
rmo_n_particles = 50
n_dimensions = 20
rmo_wMax = 1 
rmo_wMin = 0
rmo_xmin = -30
rmo_xmax = 30
rmo_particles_pos = []
pso_iteration =  15
def RadialMove(particle_center):
    for i in range(0 , rmo_n_particles):
        pos = []
        for  j in range(0 , n_dimensions ):
            val = rmo_xmin + np.random.uniform(0 , 1) *(rmo_xmax - rmo_xmin)
            pos.append(val)
        rmo_particles_pos.append(pos)
    center = np.zeros((1 , n_dimensions ))[0]
    f =fun(rmo_particles_pos[0])
    Optimum = fun(rmo_particles_pos[0])
    gbest = np.array(rmo_particles_pos[0])
    for gen in range(0 , nb_generation):
        ineteria = rmo_wMax - i *((rmo_wMax - rmo_wMin) / float(nb_generation))
        velocities =  []
        for i in range(0 , rmo_n_particles ):
            vel = []
            for j in range(0 , n_dimensions):
                r1 =  np.random.uniform(0 , 1)
                vmax = float((rmo_xmax - rmo_xmin ) / 7)
                velocity = r1 *  vmax
                vel.append(velocity)
            velocities.append(vel)
        for i in range(0 ,  rmo_n_particles):
            for j in range(0,n_dimensions) :
                rmo_particles_pos[i][j] = ineteria*velocities[i][j] + particle_center[j]
                if(rmo_particles_pos[i][j] > rmo_xmax ):
                    rmo_particles_pos[i][j] = rmo_xmax
                if(rmo_particles_pos[i][j] <  rmo_xmin):
                    rmo_particles_pos[i][j] = rmo_xmin
        scores = []
        temp = np.zeros((1 , n_dimensions))[0]
        for i in range(0 ,rmo_n_particles ):
            out = fun(rmo_particles_pos[i])
            scores.append(out)
            if out < f :
                f = out 
                pb = i
                RbestLoc = rmo_particles_pos[i]
                if(f < Optimum):
                    Optimum = f
                    gbest = RbestLoc
                    temp = np.ones((1 , n_dimensions))[0]
                    for t in range(0 , n_dimensions):    
                        temp[t] = RbestLoc[t]
        for t in range(0 , n_dimensions):    
            gbest[t] = temp[t]
    return Optimum    , gbest
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

  def update_velocity(self, x, v, p_best, g_best, c0=1.5, c1=1.5, w=0.75):
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

  def optimize(self, maxiter=pso_iteration):
    """
      Run the PSO optimization process untill the stoping criteria is met.
      Case for minimization. The aim is to minimize the cost function.
      Args:
          maxiter (int): the maximum number of iterations before stopping
                         the optimization.
      Returns:
          The best solution found (array-like).
    """
    for t in range(maxiter):
      for i in range(self.n_particles):
          x = self.particles_pos[i]
          v = self.velocities[i]
          p_best = self.p_best[i]
          self.velocities[i] = self.update_velocity(x, v, p_best, self.g_best)
          self.particles_pos[i] = self.update_position(x, v)
          # Update the best position for particle i
          rmo_Optimum , rmo_gbest  = RadialMove(self.particles_pos[i])   
          if rmo_Optimum < self.func(p_best):
              self.p_best[i] = rmo_gbest
          # Update the best position overall
          if rmo_Optimum < self.func(self.g_best):
              self.g_best = rmo_gbest
    return self.g_best, self.func(self.g_best)
start_time = time.time()
init_pos = np.ones(n_dimensions)
PSO_s = PSO(func=fun, init_pos=init_pos, n_particles=150)
res_s = PSO_s.optimize()
print(">>>>"  , res_s[1])
print(">>>>"  , res_s[0])    
print("--- %s seconds ---" % (time.time() - start_time))
             
            
            