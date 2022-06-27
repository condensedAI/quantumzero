import numpy as np
from qutip import *
from qutip.piqs import jspin
from numpy import *

class AnalogAnnealer():
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
        NL = self.T/self.dt
        # Create a grid of linearly spaced time points
        t = np.linspace(0, self.T, int(NL+1))

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

class DigitalAnnealer():
    def __init__(self, num_qubits, n_trotter_steps, n_frequency_components, H0, Hf, psi0, psif):
        self.P = n_trotter_steps
        self.H0 = H0
        self.Hf = Hf
        self.psi0 = psi0
        self.psif = psif
        self.num_qubits = num_qubits
        self.N=2**num_qubits
        self.n_frequency_components = n_frequency_components
        self.sin_matrix, self.cos_matrix = self.fourier_transform(self.P, self.n_frequency_components)


    def fourier_transform(self, num_trotter_step, num_frequency_components):
        '''
        creates two matrices to derive the variational paramters gamma and beta from the fourier components
        ref. Zhou, Lukin et al PRX2021
        '''
        p_ind=(np.arange(1,num_trotter_step+1).reshape([num_trotter_step,1])-0.5)*np.pi/num_trotter_step
        q_ind=np.arange(1,num_frequency_components+1).reshape([1,num_frequency_components])-0.5

        cos_matrix = np.cos(np.dot(p_ind,q_ind))
        sin_matrix = np.sin(np.dot(p_ind,q_ind))

        return sin_matrix, cos_matrix

    def uv2schedule(self,u,v, optimization_space='frequency'):
        if optimization_space =='frequency':
            gamma = np.dot(self.sin_matrix,u).reshape(-1)
            beta = np.dot(self.cos_matrix,v).reshape(-1)
        else:
            gamma = u
            beta = v
        return gamma, beta

    def x_rotation(self, beta):
        '''
        Perform a rotation of the angle beta along the x axis.
        '''
        single_x_rot = qeye(2)*np.cos(0.5*beta) + 1j*sigmax()*np.sin(0.5*beta)
        op_list = [single_x_rot for nq in range(self.num_qubits)]

        op_x_rotation = tensor(op_list)

        return Qobj(op_x_rotation.data.reshape([self.N,self.N]))

    def digital_evo(self, gamma, beta, get_gradient=False):
        '''
        Perform the discrete time evolution of the state psi0 with n_trotter_steps steps. The Hamiltonians used for the evolution are known from the class
        Parameters:

        gamma: n_trotter_steps dimensional array with the time steps for the evolution with Hf
        beta: n_trotter_steps dimensional array with the time steps (rotations) for the evolution wit H0
        get_gradient: if True, computes the derivative of the energy with respect to the variational parameters
        '''
        assert gamma.size == self.P, f"gamma does not have the correct legth" + str(gamma.size)
        assert beta.size == self.P, f"beta does not have the correct legth " +str(beta.size)

        state_m = [self.psi0]
        for m in range(self.P):
            zphase = -1j*gamma[m]*self.Hf
            #xphase = -1j*beta[m]*self.H0
            #state_m.append(xphase.expm()*zphase.expm(method='sparse')*state_m[m])
            state_m.append(self.x_rotation(beta[m])*zphase.expm(method='sparse')*state_m[m])

        fidelity = []
        for state in state_m:
            c = state.overlap(self.psif.dag())
            fidelity.append(np.abs(c)**2)

        # Compute the expectation value of the final Hamiltonian at all intermediate steps
        energy=expect(self.Hf, state_m)

        if get_gradient:
            gamma_grad = []
            beta_grad = []
            Sx = Qobj(jspin(self.num_qubits,op='x',basis='uncoupled').data.reshape([self.N,self.N]))

            state_Cm=[self.Hf*state_m[-1]]
            for m in range(self.P):
                zphase = 1j*gamma[-1-m]*self.Hf
                state_Cm.append(zphase.expm(method='sparse')*self.x_rotation(-beta[-1-m])*state_Cm[m])

            for m in range(self.P):
                gamma_grad.append(state_Cm[self.P-m].overlap(self.Hf*state_m[m]))
                beta_grad.append(state_Cm[self.P-m-1].overlap(self.H0*state_m[m+1]))
            return 2*np.concatenate((np.array(gamma_grad).imag, np.array(beta_grad).imag))

        else:
            return energy, fidelity
