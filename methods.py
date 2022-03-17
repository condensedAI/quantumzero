import math
from numpy import *
from scipy import linalg
from scipy.optimize import minimize
import numpy as np

import random
from qutip import * 
from annealer import *

import mcts as mod_mcts

def linear(num_frequency_components, T, HB, HP, psi0, psif):
    annealer = Annealer(0.5, T, HB, HP, psi0, psif)   
    
    solution = np.array(np.zeros(num_frequency_components))
    energy, fidelity =  annealer.anneal(solution)
    return energy,fidelity


def StochasticDescent(num_qubits, num_frequency_components, T, HB,HP,psi0,psif):
    annealer = Annealer(0.5, T, HB,HP,psi0,psif)

    delta=0.01#/4
    iterNum=100
    ncan=0
    l = [x/100  for x in range(-20, 20,1)]
    
    for i in range (num_frequency_components):
        s = random.sample(l, 1) 
        obs[i]=s[0]
    print(obs)

    iter = 0
    while True:
        iter += 1

        energy, fidelity =  annealer.anneal(solution)
        ncan=ncan+1
        obs1=obs
        fid=fidelity
        num_converge = 0
        for m in range(1, 1+num_frequency_components):
            #print('Updating parameter b{}'.format(m))
            if  obs[m-1]+ delta > 0.2 or obs[m-1] - delta < -0.2:
                num_converge += 1
                continue
            obs[m-1] += delta #/m    #高频的变化幅度大于低频
            #print(delta/m )
            #print(obs)
            energy, fidelity =  annealer.anneal(solution)
            ncan=ncan+1
            if fidelity > fid:
                fid=fidelity
                #print('b{}+delta'.format(m))
                continue 
            else:
                obs[m-1] -= 2*delta #/m   
                #print(delta/m )
                #print(obs)
                energy, fidelity =  annealer.anneal(solution)     
                ncan=ncan+1
                if fidelity > fid:
                    fid=fidelity
                    #print('b{}-delta'.format(m))
                    continue 
                else:
                    obs[m-1] += delta #/m
                    num_converge += 1  
                    #print('keep invariant')  
                    continue 
        if num_converge == num_frequency_components:
            #print('fidelity:', fid,energy)
            break
        if iter > iterNum:
            print('WARNING: The algorithm does not converge')
            break 
        #print("ss:",obs)
    #print(iter)
    print("iter:",iter,"ncan:",ncan)
    return obs1, fid


def mcts(data, n_qubit,T, num_frequency_components ,H0,Hf,psi0,psif,ncandidates, cost_function_type):

    annealer = Annealer(0.5, T, H0,Hf, psi0, psif)   

    def get_reward(struct):
        delta=0.1             ##update lenth
        De=40  #20

        solution=np.zeros((num_frequency_components), dtype=np.float64)   
        for i in range(num_frequency_components):
            solution[i]=-0.2+struct[i]%De*0.01

        energy, fidelity =  annealer.anneal(solution)

        if cost_function_type == 'energy':
            cond=-energy[-1]
        elif cost_function_type == 'fidelity':
            cond=fidelity[-1]
        else:
            raise ValueError(f'wrong cost function {cost_function_type}')
        
        return cond

    myTree=mod_mcts.Tree(data,T,no_positions=5, atom_types=list(range(40)), atom_const=None, get_reward=get_reward, positions_order=list(range(5)),
            max_flag=True,expand_children=10, play_out=5, play_out_selection="best", space=None, candidate_pool_size=100,
             ucb="mean")

    res=myTree.search(display=True,no_candidates=ncandidates)

    fidelity=res.optimal_fx  
    obslist=res.optimal_candidate
 
    solution=-0.2+np.array(obslist)%40*0.01

    return solution,fidelity

def qaoa(n_qubit, num_frequency_components,num_trot_step, H0, Hf, psi0, psif, ncandidates, cost_function_type, x0=None):
    
    annealer = DigitalAnnealer(n_qubit, num_trot_step, H0, Hf, psi0, psif)

    #notation of Zhou et al. PRX 2021 for the Fourier optimization of QAOA

    p_ind=(np.arange(1,num_trot_step+1).reshape([num_trot_step,1])-0.5)*np.pi/num_trot_step
    q_ind=np.arange(1,num_frequency_components+1).reshape([1,num_frequency_components])-0.5
    cos_matrix = np.cos(np.dot(p_ind,q_ind))
    sin_matrix = np.sin(np.dot(p_ind,q_ind))

    def get_reward(x, cost_function_type='energy'):
        u = x[:num_frequency_components].reshape([num_frequency_components,1])
        v = x[num_frequency_components:].reshape([num_frequency_components,1])
        gamma = np.dot(sin_matrix,u).reshape(-1)
        beta = np.dot(cos_matrix,v).reshape(-1)
        energy, fidelity = annealer.digital_evo(gamma,beta)

        if cost_function_type == 'energy':
            return energy[-1]
        elif cost_function_type == 'fidelity':
            return 1.0-fidelity[-1]
        else:
            raise ValueError(f'wrong cost function type: {cost_function_type}') 
    if x0 is None:
        x0 = np.random.rand(2*num_frequency_components)
    res = minimize(get_reward, x0,args=(cost_function_type),  method='BFGS',options={'gtol':1e-2, 'disp': True})
 
    return res.x, res.fun, res.nfev
