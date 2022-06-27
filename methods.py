import math
from numpy import *
from scipy import linalg
from scipy.optimize import minimize
import numpy as np

import random
from qutip import *
from annealer import *

import mcts as mod_mcts

def linear(num_frequency_components, T, num_time_steps, HB, HP, psi0, psif):
    annealer = AnalogAnnealer(T/num_time_steps, T, HB, HP, psi0, psif)

    solution = np.array(np.zeros(num_frequency_components))
    energy, fidelity =  annealer.anneal(solution)
    return energy,fidelity


def StochasticDescent(num_qubits, num_frequency_components, T, HB,HP,psi0,psif):
    annealer = AnalogAnnealer(0.5, T, HB,HP,psi0,psif)

    delta=0.01#/4
    iterNum=100
    ncan=0
    l = [x/100  for x in range(-20, 20,1)]

    obs = np.zeros(num_frequency_components)
    for i in range (num_frequency_components):
        s = random.sample(l, 1)
        obs[i]=s[0]

    iter = 0
    while True:
        iter += 1

        energy, fidelity =  annealer.anneal(obs)
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
            energy, fidelity =  annealer.anneal(obs)
            ncan=ncan+1
            if fidelity > fid:
                fid=fidelity
                #print('b{}+delta'.format(m))
                continue
            else:
                obs[m-1] -= 2*delta #/m
                #print(delta/m )
                #print(obs)
                energy, fidelity =  annealer.anneal(obs)
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

def reward_function(x, annealer, annealing_type, num_frequency_components, cost_function_type='energy', optimization_space='frequency', delta=0.01, De=40):

    if annealing_type=='analog':
        solution=np.zeros((num_frequency_components), dtype=np.float64)
        for i in range(num_frequency_components):
            solution[i]= -0.2+x[i]%De * delta

        energy, fidelity = annealer.anneal(solution)

    elif annealing_type =='digital':
        if optimization_space=='frequency':
            u = np.array(x[:num_frequency_components]).reshape([num_frequency_components,1])
            v = np.array(x[num_frequency_components:]).reshape([num_frequency_components,1])
            gamma, beta = annealer.uv2schedule(u,v)

        elif optimization_space=='t_schedule':
            gamma=np.array(x[:annealer.P])
            beta=np.array(x[annealer.P:])

        else:
            raise ValueError(f'Unknown optimization space: {optimization_space}')

        energy, fidelity = annealer.digital_evo(gamma,beta)
    else:
        raise ValueError(f'Uknown annealing type: {annealing_type}')

    if cost_function_type == 'energy':
        cond = -energy[-1]
    elif cost_function_type == 'fidelity':
        cond = fidelity[-1]
    else:
        raise ValueError(f'Unknown cost function type: {cost_function_type}')

    return cond

def mcts(n_qubit, T, num_frequency_components, num_trotter_steps, H0, Hf, psi0, psif, ncandidates, cost_function_type='energy', annealing_type='analog', optimization_space='frequency'):

    if annealing_type=='analog':
        annealer = AnalogAnnealer(T/num_trotter_steps, T, H0, Hf, psi0, psif)
        n_search = num_frequency_components
    elif annealing_type=='digital':
        annealer = DigitalAnnealer(n_qubit, num_trotter_steps, num_frequency_components, H0, Hf, psi0, psif)
        if optimization_space =='frequency':
            n_search = 2*num_frequency_components
        else:
            n_search = 2*num_trotter_steps
    else:
        raise ValueError(f'Unknown annealing type: {annealing_type}')

    def get_reward(struct):
        cond = reward_function(struct, annealer, annealing_type, num_frequency_components,
                                cost_function_type=cost_function_type, optimization_space=optimization_space)
        return cond

    mctree = mod_mcts.Tree(no_positions=n_search,
                                atom_types=list(range(40)),
                                atom_const=None,
                                get_reward=get_reward,
                                positions_order=list(range(n_search)),
                                max_flag=True,
                                expand_children=10,
                                play_out=5,
                                play_out_selection="best",
                                space=None,
                                candidate_pool_size=100,
                                ucb="mean")

    solution, fidelity = mctree.search(display=True, no_candidates=ncandidates)

    solution = -0.2+np.array(solution)%40*0.01

    return solution, fidelity

def mcts_v2(n_qubit, T, num_frequency_components, num_trotter_steps, H0, Hf, psi0, psif, ncandidates, cost_function_type='energy', annealing_type='analog', optimization_space='frequency'):

    if annealing_type=='analog':
        annealer = AnalogAnnealer(T/num_trotter_steps, T, H0, Hf, psi0, psif)
        n_search = num_frequency_components
    elif annealing_type=='digital':
        annealer = DigitalAnnealer(n_qubit, num_trotter_steps, num_frequency_components, H0, Hf, psi0, psif)
        if optimization_space =='frequency':
            n_search = 2*num_frequency_components
        else:
            n_search = 2*num_trotter_steps
    else:
        raise ValueError(f'Unknown annealing type: {annealing_type}')

    def get_reward(struct):
        cond = reward_function(struct, annealer, annealing_type, num_frequency_components,
                                cost_function_type=cost_function_type, optimization_space=optimization_space)
        return cond

    mctree = mod_mcts.Tree(no_positions=n_search,
                                atom_types=list(range(40)),
                                atom_const=None,
                                get_reward=get_reward,
                                positions_order=list(range(n_search)),
                                max_flag=True,
                                expand_children=10,
                                play_out=5,
                                play_out_selection="best",
                                space=None,
                                candidate_pool_size=100,
                                ucb="mean")

    solution, fidelity = mctree.find_best_candidate(ncandidates)
    solution = -0.2+np.array(solution)%40*0.01

    return solution, fidelity

def qaoa(n_qubit, num_frequency_components, num_trot_step, H0, Hf, psi0, psif, ncandidates, cost_function_type='energy', x0=None, optimization_space='frequency',
        jacobian=False):
    '''
    Perform QAOA, either optimizing directly the digitized annealing schedule with num_trot_steps components or in the Fourier components
    Parameters:
        n_qubit: number of variables in the 3-SAT problem

        num_frequency_components: number of frequency compnents of the digitized annealing schedule. The imension of the search space is 2*num_frequency_components

        num_trot_step: number of trotter steps in the digitize annealing. The schedule is defined by 2*num_trot_steps real numbers

        H0, psi0: initial (mixing) Hamiltonian and initial state

        Hf, psif: problem Hamiltonian and target ground state

        ncandidates: number of local minimum search

        cost_function_type: <energy> or <fidelity>, determines whch cost function is used in thr qaoa

        x0: starting point for the local minimum search. if None, it is an array of random numbers

        optimization_space: <frequency> or <t_schedule>, determines whether the local minimum search is performed in the frequency or time domain

    Returns:
        solution: list with the ncandidate minima found
        reward: list with the ncandidate minimized cost function
        num_queries: list with the number of function evaluation for each local search

    '''
    annealer = DigitalAnnealer(n_qubit, num_trot_step,num_frequency_components, H0, Hf, psi0, psif)
    if cost_function_type == 'fidelity':
        jacobian=False

    if optimization_space =='frequency':
        n_search = num_frequency_components

    elif optimization_space=='t_schedule':
        n_search = num_trot_step
    else:
        raise ValueError(f'wrong optimization space: {optimization_space}')

    def get_reward(x, cost_function_type='energy', optimization_space='frequency'):
        if optimization_space=='frequency':
            u = x[:num_frequency_components].reshape([num_frequency_components,1])
            v = x[num_frequency_components:].reshape([num_frequency_components,1])
            gamma, beta = annealer.uv2schedule(u,v)

        elif optimization_space=='t_schedule':
            gamma=x[:num_trot_step]
            beta=x[num_trot_step:]

        else:
            raise ValueError(f'Unknown optimization space: {optimization_space}')

        energy, fidelity = annealer.digital_evo(gamma,beta)

        if cost_function_type == 'energy':
            return energy[-1]
        elif cost_function_type == 'fidelity':
            return 1.0-fidelity[-1]
        else:
            raise ValueError(f'Unknown cost function type: {cost_function_type}')

    def get_gradient(x, cost_function_type='energy', optimization_space='frequency'):

        if optimization_space=='frequency':
            u = x[:num_frequency_components].reshape([num_frequency_components,1])
            v = x[num_frequency_components:].reshape([num_frequency_components,1])

            gamma, beta = annealer.uv2schedule(u,v)
            grad_gb = annealer.digital_evo(gamma,beta,get_gradient=True)
            grad_u = np.dot(grad_gb[:gamma.size],annealer.sin_matrix)
            grad_v = np.dot(grad_gb[gamma.size:],annealer.cos_matrix)
            grad_E = np.concatenate((grad_u,grad_v))

        elif optimization_space == 't_schedule':
            gamma=x[:num_trot_step]
            beta=x[num_trot_step:]

            grad_E = annealer.digital_evo(gamma,beta,get_gradient=True)

        return grad_E

    solution = []
    reward = []
    num_queries = []
    for n_opt in range(ncandidates):
        if x0 is None:
            x0 = np.random.rand(2*n_search)
        else:
             x0+=np.random.rand(2*n_search)*0.1

        if jacobian:
            res = minimize(get_reward, x0, jac=get_gradient, args=(cost_function_type, optimization_space),  method='BFGS',options={'gtol':1e-2, 'disp': True})
        else:
            res = minimize(get_reward, x0,args=(cost_function_type, optimization_space),  method='BFGS',options={'gtol':1e-2, 'disp': True})
        solution.append(res.x)
        reward.append(res.fun)
        num_queries.append(res.nfev)

    best_result = np.argmin(np.array(reward))
    return solution[best_result], reward[best_result], np.array(num_queries).sum()
