import numpy as np 
from qutip import * 
from numpy import *

class Annealer():
    def __init__(self, dt, T, H0, Hf, psi0, psif):
        self.T = T
        self.dt = dt
        
        self.psi0 = psi0
        self.psif = psif

        self.H0 = H0
        self.Hf = Hf

    def s(self, t, args):
        s = t/self.T 
        for j,b in enumerate(args):
            f = (j+1)*np.pi*t/self.T
            s += args[b] * np.sin(f)
        return s

    def anneal(self, bstate):
        args = {}
        for i, b in enumerate(list(bstate)):
            args['b{}'.format(i+1)] = b 
            
        # The total number of steps
        NL=self.T/self.dt
        # Create a grid of linearly spaced time points
        t = np.linspace(self.dt, self.T, int(NL))
        
        # Perform the full time evolution
        H = [[self.H0, lambda t,args : 1 - self.s(t,args)], [self.Hf, self.s]]
        output = mesolve(H, self.psi0, t, args=args)
        
        # Compute the fidelity in all intermediate steps defined by the array t      
        fidelity = []
        for state in output.states:
            c = state.overlap(self.psif.dag())
            fidelity.append(np.abs(c)**2)

        # Compute the expectation value of the final Hamiltonian at all intermediate steps
        energy=expect(self.Hf, output.states)
    
        return energy, fidelity
