import numpy as np 
from qutip import * 
from numpy import *

class Annealer():
    def __init__(self, dt, T, H0, Hf, psi0, psif):
        self.T = T
        self.dt = 0.5
        
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
            
        # Define the step size
        dt=0.5
        # The total number of steps
        NL=self.T/dt
        # Create a grid of linearly spaced time points
        t = np.linspace(dt, self.T-dt, int(NL))
        
        # Perform the full time evolution
        H = [[self.H0, lambda t,args : 1 - self.s(t,args)], [self.Hf, self.s]]
        output = mesolve(H, self.psi0, t, args=args)
      
        # Take the final state...
        final_state = output.states[-1]
        # ... and compute the overlap with the perfect final state ...
        c = (self.psif.dag()) * final_state    ####overlap
        fidelity = np.abs(c[0,0])**2
        # ... and its energy wrt the final Hamiltonian
        x = final_state.dag() * self.Hf * final_state      ###energy
        energy= x[0,0]
    
        return energy, fidelity