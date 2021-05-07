# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 00:08:13 2021

@author: hassa
"""
import numpy as np
import time
import math
number_variables = 10
upper_bownd = np.array([10,10,10,10,10,10,10,10,10,10])
lower_bownd = np.array([-10,-10,-10,-10,-10,-10,-10,-10,-10, -10])
import random
import matplotlib.pyplot as plt
def SphereFunction(x):
    sum =0
    for i in range(0 , len(x)):
        sum += x[i]
    return sum
number_particles = 30
maxIteration = 500
wMax = 0.8
wMin = 0.2
c1 = 2.05
c2 = 2.05
threshold = 100
alpha = 1
betta = 2

def CostFunction(particle_objective_value , particle_new_objective_value , g_best ):
    cost = (particle_objective_value - g_best) / (particle_new_objective_value - g_best)
    return cost    

def InitializeCostMatrix(particles):
    values = []
    counter = 0
    for particle in particles:
        particle_value = []
        for i in range(0 , len(particles)):
            if i == counter :
                particle_value.append(0)
            else:
                particle_value.append(1)
        counter+= 1
        values.append(particle_value)
    matrix = np.matrix(values,  dtype = float )
    return matrix
def InitializeDesirabilityMatrix(particles):
    values = []
    counter = 0
    for particle in particles:
        particle_value = []
        for i in range(0 , len(particles)):
            if i == counter :
                particle_value.append(0)
            else:
                particle_value.append(1)
        counter+= 1
        values.append(particle_value)
    matrix = np.matrix(values ,  dtype = float)
    return matrix
def Probability(current_node , tour):
    P_allNodes = desirability_matrix**betta
    for i in tour:
        P_allNodes[current_node , i] = 0
    P_allNodes = P_allNodes / sum(P_allNodes)
    return P_allNodes[current_node]
toplogies = []
def CreateRoute(phi):
    route = Route([])
    route.topologies = []
    topology = Topology([] , phi)
    initiale_particle = np.random.randint(0 ,number_particles )
    topology.tour = [initiale_particle]
    for j in range(1 , phi):
        current_node = topology.tour[len(topology.tour) - 1]
        p = Probability(current_node , topology.tour)
        next_node = RouletteWheel(p)
        topology.tour = np.append(topology.tour , next_node)
        route.topologies = np.append(route.topologies , topology)
        toplogies.append(topology.tour)
    return route
def RouletteWheel(P):
    cumsum = np.cumsum(P , dtype=float)
    random_number =   random.uniform(0, 1)
    rand =  np.where(random_number <= cumsum)[1]
    counter = 0
    while(len(rand) < 1):
         random_number =   random.uniform(0, 1)
         rand =  np.where(random_number <= cumsum)[1]
         counter += 1
         if (counter == threshold):
             return 0
             break
    rand = rand[0]
    next_node =  rand
    return next_node

class Route :
    def __init__(self , topologies):
        self.topologies = topologies
class Topology :
    def __init__(self , tour , fitness):
        self.tour = tour
class Swarm :
    def __init__(self , particles , g_best):
        self.particles = particles
        self.g_best = g_best

class Particle:
    def __init__(self ,  position , velocity , p_best , objective_value):
        self.position = position
        self.velocity = velocity
        self.p_best = p_best
        self.objective_value = objective_value
        
class PBest:
    def __init__(self , position , objective_value):
        self.position = position
        self.objective_value = objective_value
class GBest:
    def __init__(self , position , objective_value):
        self.position = position
        self.objective_value = objective_value
        
best_values = np.array([])
particles = np.array([])
for i in range(0 , number_particles):
    random_position = np.random.randint(np.max(upper_bownd) , size=(len(upper_bownd)))
    position = (upper_bownd - lower_bownd)* random_position + lower_bownd
    velocity = np.zeros((len(upper_bownd),), dtype=int)
    personal_best = PBest(np.zeros((len(upper_bownd),), dtype=int) , math.inf)
    particle = Particle(position , velocity , personal_best , math.inf)
    particles =np.append(particles,particle)


def Impact(my_swarm , index , number_particles):
      velocity = ineteria*my_swarm.particles[index].velocity + c2 * random.uniform(0, 1) * (my_swarm.g_best.position - my_swarm.particles[index].position)
      for t in range(0 , number_particles):
          velocity += c1 * random.uniform(0, 1) * (my_swarm.particles[index].position - my_swarm.particles[t].position)
          position =  my_swarm.particles[index].position + velocity
          cost = CostFunction(my_swarm.particles[index].objective_value , SphereFunction(position) , my_swarm.g_best.objective_value )
          cost_matrix[index , t] = abs(cost)
          desirability_matrix[index , t] = 1/abs(cost)

start_time = time.time()

my_swarm = Swarm(particles , GBest(np.zeros((len(upper_bownd),), dtype=int) , math.inf))
cost_matrix = InitializeCostMatrix(my_swarm.particles)


desirability_matrix = InitializeDesirabilityMatrix(my_swarm.particles)
iterations =  np.array([])
for i in range(0 , maxIteration):
    for j in range(0 , number_particles):
        my_swarm.particles[j].objective_value = SphereFunction(my_swarm.particles[j].position)
        if my_swarm.particles[j].objective_value < my_swarm.particles[j].p_best.objective_value:
            my_swarm.particles[j].p_best.position = my_swarm.particles[j].position
            my_swarm.particles[j].p_best.objective_value = my_swarm.particles[j].objective_value
        if my_swarm.particles[j].objective_value < my_swarm.g_best.objective_value:
            my_swarm.g_best.position = my_swarm.particles[j].position
            my_swarm.g_best.objective_value = my_swarm.particles[j].objective_value
    ineteria = wMax - i *((wMax - wMin) / maxIteration)
    for k in range(0 , number_particles):
        Impact(my_swarm , k , number_particles)
        phi = np.random.randint(0 , number_variables - 1)
        route = CreateRoute(phi)     
        my_swarm.particles[k].velocity = ineteria*my_swarm.particles[k].velocity + c2 * random.uniform(0, 1) * (my_swarm.g_best.position - my_swarm.particles[k].position)
        for topology in route.topologies:
            for selected_particule in topology.tour:
                if selected_particule != 0:
                    my_swarm.particles[k].velocity += c1 * random.uniform(0, 1) * (my_swarm.particles[k].position - my_swarm.particles[selected_particule].position)
        my_swarm.particles[k].position =  my_swarm.particles[k].position +  my_swarm.particles[k].velocity
    best_values = np.append(best_values , my_swarm.g_best.objective_value)
    iterations = np.append(iterations , i)
plt.plot(iterations , best_values)
print(best_values)
plt.title("Weight Versus Iteration" , fontsize=20 , fontweight = 'bold')
plt.xlabel("Iterations" , fontsize=20 , fontweight = 'bold')
plt.ylabel("Weight" , fontsize=20 , fontweight = 'bold')
plt.show()

print("--- %s seconds ---" % (time.time() - start_time))
