import math
from scipy import linalg
import numpy as np
from problems import *
import datetime
from scipy.optimize import minimize
import methods
#from dynamics import SatSearchEfficient
from annealer import *
import argparse
import subprocess



###############################################################################

# definition of the input parameters

parser=argparse.ArgumentParser(description='Quantum annealing path optimization on 3-SAT hard instances')

parser.add_argument('Opt_type', type=str, help='<BFGS> or <MCTS>, determine which algorithm is used for the path optimization')
parser.add_argument('--n_qubit', default=7, type=int, help='Number of spin variables, default 7')
parser.add_argument('--n_instances', default=1, type=int, help='Number of 3-SAT instances solved, default 1')
parser.add_argument('--T_ann', default=10, type=float, help='Annealing time, default 10')
parser.add_argument('--Nt', default=100, type=int, help='Number of time steps at which the solution of the Scrhoedigner eq. is computed, default 100')
parser.add_argument('--Mcut', default=5, type=int, help='Number of Fourier components in the schedule optimization,  default 5')
parser.add_argument('--cost_function', default='energy', type=str, help='<energy> or <fidelity>, determines which is used as a cost function for the minimizer, default energy')
parser.add_argument('--n_candidates', default=1000, type=int, help='Number of episodes allowed in the MCTS, default 1000)')
parser.add_argument('--retrain', default=False, type=bool, help='If True, the BFGS optimization starts from the optimal paramters found in the first instance, defualt False. STILL TO BE FULLY IMPLEMENTED')


args = parser.parse_args()

opt_type = args.Opt_type
assert opt_type=='BFGS' or opt_type=='MCTS'

n_qubit = args.n_qubit
n_instances = args.n_instances 
T = args.T_ann
Nt = args.Nt
Mcut = args.Mcut
cost_function_type = args.cost_function
n_candidates = args.n_candidates
retrain = args.retrain

filename='dataset/sat'+str(n_qubit)+'.txt'
dataset = np.loadtxt(filename, dtype='int')

if dataset.shape[0] < n_instances:
    n_instances = dataset.shape[0]

###########################################################################

# DEfinition of the functions used by the scipy minimizer

def cost_function(x):
    en, fid = pathdesign.anneal(x)
    if cost_function_type == 'energy':    
        return np.real(en)[-1]
    elif cost_function_type == 'fidelity':
        return 1-fid[-1]
    else:
        raise ValueError(f'Wrong cost function {cost_function_type}, exit')

##############################################################################

if opt_type=='BFGS':
    header = 'Gradient-based optimization of the annealing schedule\n'
elif opt_type=='MCTS':
    header = 'MCTS optimization of the annealing schedule\n'

header += 'n_qubit=' + str(n_qubit) + '\n' + 'T_ann=' + str(T) + '\n' + 'Mcut=' + str(Mcut) + '\n'
header += 'Cost function: '+ cost_function_type + '\n'
header += 'Initialize parameters from previous instance: '+ str(retrain)
print(header)
###########################################################################

# Running the optimization of the QA path  


best_result=[]
best_path=[]

bzero = np.zeros(Mcut) 
dt=T/Nt
time_vec = np.linspace(dt, T, Nt)
tevo=[]
# for the moment we start from the linear annealing schedule. In future developments, might be interesting to average over small noise

for instance in range(n_instances):
    
    # definition of the problem and mixing Hamiltonian (should be improved to avoid uselss repetitions of the same operation) 
    result=dataset[instance,:]
    #HB,HP,psi0,psif=system.satSystem(n_qubit,result)
    H0,Hf,psi0,psif=create_3SAT_H_and_psi(n_qubit,result)
   
    # creation of the SAT class that contains the method for time evolution
    #pathdesign=SatSearchEfficient(n_qubit,T,Mcut,HB,HP,psi0,psif,dt)
    pathdesign=Annealer(dt, T, H0, Hf, psi0, psif)

    # optimization of the FOurier component. For the moment the minimization algorithm is fixed, we might want to leave it as an external param
    if opt_type == 'BFGS': 
        if instance > 0 and retrain:
            res=minimize(cost_function, obs, method='BFGS', options={'gtol':1e-2, 'disp':True} )
        else:
            res=minimize(cost_function, bzero, method='BFGS', options={'gtol':1e-2, 'disp':True} )

        obs = res.x
        nfev=res.nfev    

    elif opt_type == 'MCTS':
        obs, fid = methods.mcts(1,n_qubit,T,Mcut,H0,Hf,psi0,psif,n_candidates, cost_function_type)
        obs=obs.reshape(-1)
        nfev=n_candidates 
        
    en, fid = pathdesign.anneal(obs)

    # linear schedule comparison
    en_lin, fid_lin = pathdesign.anneal(bzero)
    print(f'energy={en[-1]}, fidelity={fid[-1]}, improvement over linear={fid[-1]/fid_lin[-1]}\n')
    print(f'optimal F components={obs}\n')
    best_result.append([instance, fid[-1],en[-1], nfev, fid_lin[-1], en_lin[-1]])
    best_path.append(obs)    
    tevo.append(np.concatenate((time_vec.reshape([Nt,1]), np.array(fid).reshape([Nt,1]), np.array(en).reshape([Nt,1]),
                                np.array(fid_lin).reshape([Nt,1]), np.array(en_lin).reshape([Nt,1])),axis=1))
    
tevo=np.array(tevo).reshape([Nt*n_instances,5])

############################################################################

# Saving files

outdir = '../Output/'+opt_type+'/retrain'+str(retrain)+'/N' + str(n_qubit) + '/'
# This line works only on unix systems
subprocess.run(['mkdir', '-p', outdir])

header_results = header+'\n' + '1-instance, 2-fidelity, 3-energy, 4-n_fev, 5-lin. fidelity, 6-lin. energy'

header_tevo = header+'\n' + '1-time, 2-fidelity, 3-energy, 4-lin. fidelity, 4-lin. energy' 

filename_results = '3SAT-QAperformance_N'+str(n_qubit)+'_T'+str(T)+'_Mcut'+str(Mcut) + '_cf-'+cost_function_type +'.dat'

filename_tevo = '3SAT-QAtevo_N'+str(n_qubit)+'_T'+str(T)+'_Mcut'+str(Mcut) + '_cf-'+cost_function_type +'.dat'

filename_path = '3SAT-QApath_N'+str(n_qubit)+'_T'+str(T)+'_Mcut'+str(Mcut) + '_cf-'+cost_function_type +'.dat'

np.savetxt(outdir+filename_results, np.array(best_result), header=header_results)
np.savetxt(outdir+filename_tevo, tevo, header=header_tevo)
np.savetxt(outdir+filename_path, np.array(best_path), header=header)


