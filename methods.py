import math
from numpy import *
from scipy import linalg
import numpy as np

import random
from qutip import * 
from annealer import Annealer

import mcts

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


def mcts(data, n_qubit,T, num_frequency_components ,HB,HP,psi0,psif,ncandidates):

    annealer = Annealer(0.5, T, HB,HP,psi0,psif)   

    def get_reward(struct):
        delta=0.1             ##update lenth
        De=40  #20

        obs=np.zeros((num_frequency_components), dtype=np.float64)   
        for i in range(num_frequency_components):
            obs[i]=-0.2+struct[i]%De*0.01

        energy, fidelity =  annealer.anneal(solution)
        cond=fidelity  

        return cond

    myTree=mcts.Tree(data,T,no_positions=5, atom_types=list(range(40)), atom_const=None, get_reward=get_reward, positions_order=list(range(5)),
            max_flag=True,expand_children=10, play_out=5, play_out_selection="best", space=None, candidate_pool_size=100,
             ucb="mean")

    res=myTree.search(display=True,no_candidates=ncandidates)

    fidelity=res.optimal_fx  
    obs=res.optimal_candidate 

    return obs,fidelity