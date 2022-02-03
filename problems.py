import numpy as np
from qutip import * 

def create_3SAT_H_and_psi(num_qubits,result):
    N = 2**num_qubits
    
    # The initial transverse field Hamiltonian
    single_qubit_h = 0.5*(identity(2) - sigmax())
    H_initial = 0
    for i in np.arange(num_qubits):
        operator_list = [single_qubit_h if j == i else identity(2) for j in range(num_qubits)]
        H_initial = H_initial + tensor(operator_list)
    H_initial = Qobj(H_initial.data.reshape(128,128))        
        
    # The initial state
    psi_initial = Qobj( np.sqrt(1/N) * np.ones(N) )
    
    # The final Hamiltonian, whose diagonal entries simply count how many times 
    # the corresponding (binary counted) basis state violates clauses
    counts = np.bincount(result)
    H_final = Qobj(np.diag(counts, k=0))
    
    # The corresponding lowest energy eigenstate is that basis state that violates no clauses,
    # meaning the one for which the diagonal entry in H_final is a 0.
    basis_state_not_violating_any_clause = np.where(counts == 0)[0][0]
    # The final state just has a 1 at that index
    psi_final = basis(N, basis_state_not_violating_any_clause)

    return H_initial, H_final, psi_initial, psi_final