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
    H_initial = Qobj(H_initial.data.reshape(N,N))        
        
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

def create_MaxCut_H_and_psi(num_qubits,graph):
    N = 2**num_qubits

    # The initial transverse field Hamiltonian
    single_qubit_h = 0.5*(identity(2) - sigmax())
    H_initial = 0
    for i in np.arange(num_qubits):
        operator_list = [single_qubit_h if j == i else identity(2) for j in range(num_qubits)]
        H_initial = H_initial + tensor(operator_list)
    H_initial = Qobj(H_initial.data.reshape(N,N))        
        
    # The initial state
    psi_initial = Qobj( np.sqrt(1/N) * np.ones(N) )

    # final Hamiltonian; graph should be the NxN connectivity matrix of the graph
    hdiag=[]
    for b in range(N):
        basis_vec=2*int2bin(b,num_qubits)-1
        hdiag.append((np.dot(basis_vec,np.dot(graph,basis_vec))+num_qubits*3)/4)
    H_final = Qobj(np.diag(hdiag, k=0))
    
    basis_states_maxcut = np.where(hdiag==min(hdiag))[0]
    # the classical ground state will be degenerate in general; let us bild a uniform superposition
    psi_final = Qobj()
    for basis_index in basis_states_maxcut:
        psi_final += basis(N,basis_index)

    psi_final /= np.sqrt(len(basis_states_maxcut))
    return H_initial, H_final, psi_initial, psi_final

def int2bin(n, nbit):
    assert 2**nbit > n
    binstr = format(n,f'0{nbit}b')
    bin_rep=[int(i) for i in binstr]
    
    return np.array(bin_rep)

def convert(filename, n, k=3):
    # reads the scd file and tranform the output in a connectivity matrix
    num_edges = int(n*k/2)
    f = open(filename, "r")
    values = np.fromfile(f, dtype=np.uint8)
    read_values = 0
    code = []
    while read_values < len(values):
        # dekomp(file,code)
        samebits = values.item(read_values)
        read_values += 1
        readbits = num_edges - samebits
        code = code[:samebits] + list(values[read_values:read_values+readbits])
        read_values += readbits
        # codetonlist(code,l)
        graph = np.zeros((n, n), dtype=np.uint8)
        v = 0
        count = [0] * n
        for w in code:
            w -= 1  # We are indexing from 0
            while(count[v] == k):
                v += 1
            # edge (v, w)
            graph.itemset((v, w), 1)
            graph.itemset((w, v), 1)
            count[v] += 1
            count[w] += 1
        yield graph


