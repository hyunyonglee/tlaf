import numpy as np
from tenpy.networks.mps import MPS
from tenpy.algorithms import dmrg, tebd
import argparse
import logging.config
import os
import os.path
import h5py
from tenpy.tools import hdf5_io
import model


def ensure_dir(f):
    d=os.path.dirname(f)
    if not os.path.exists(d):
        os.makedirs(d)
    return d



def measurements(psi):
    # Entanglement entropy
    EE = psi.entanglement_entropy()

    # Local observables
    Sx = psi.expectation_value("Sx")
    Sy = psi.expectation_value("Sy")
    Sz = psi.expectation_value("Sz")
    
    # System size
    L = psi.L

    # Total <S_alpha^2>
    Sx2 = 0.0
    Sy2 = 0.0
    Sz2 = 0.0
    
    # Loop over i <= j and exploit symmetry
    for i in range(L):
        for j in range(i, L):
            weight = 2.0 if i != j else 1.0  # symmetry factor
            Sx2 += weight * np.real(
                psi.expectation_value_term([("Sx", i), ("Sx", j)])
            )
            Sy2 += weight * np.real(
                psi.expectation_value_term([("Sy", i), ("Sy", j)])
            )
            Sz2 += weight * np.real(
                psi.expectation_value_term([("Sz", i), ("Sz", j)])
            )
            
    # Total <S_alpha>
    Sx_total = np.sum(Sx)
    Sy_total = np.sum(Sy)
    Sz_total = np.sum(Sz)

    # Variances: Var(S_alpha) = <S_alpha^2> - <S_alpha>^2
    Sx_var = Sx2 - Sx_total**2
    Sy_var = Sy2 - Sy_total**2
    Sz_var = Sz2 - Sz_total**2
    Sp_var = Sx_var + Sy_var + Sz_total

    Sx = np.real(Sx)
    Sy = np.real(Sy)
    Sz = np.real(Sz)
        
    return EE, Sx, Sy, Sz, Sx_var, Sy_var, Sz_var, Sp_var


def write_data( psi, E, EE, Sx, Sy, Sz, Sx_var, Sy_var, Sz_var, Sp_var, Lx, Ly, Jxx, hz, path ):

    ensure_dir(path+"/observables/")
    ensure_dir(path+"/mps/")

    data = {"psi": psi}
    with h5py.File(path+"/mps/psi_Lx_%d_Ly_%d_Jxx_%.2f_hz_%.2f.h5" % (Lx, Ly, Jxx, hz), 'w') as f:
        hdf5_io.save_to_hdf5(f, data)

    file_EE = open(path+"/observables/EE.txt","a", 1)    
    file_Sx = open(path+"/observables/Sx.txt","a", 1)
    file_Sy = open(path+"/observables/Sy.txt","a", 1)
    file_Sz = open(path+"/observables/Sz.txt","a", 1)
        
    file_EE.write(f"{Jxx} {hz}  {'  '.join(map(str, EE))}\n")
    file_Sx.write(f"{Jxx} {hz}  {'  '.join(map(str, Sx))}\n")
    file_Sy.write(f"{Jxx} {hz}  {'  '.join(map(str, Sy))}\n")
    file_Sz.write(f"{Jxx} {hz}  {'  '.join(map(str, Sz))}\n")
    
    file_EE.close()
    file_Sx.close()
    file_Sy.close()
    file_Sz.close()
    
    #
    file = open(path+"/observables.txt","a", 1)    
    file.write(f"{Jxx} {hz} {E} {np.max(EE)} {np.mean(Sx)} {np.mean(Sy)} {np.mean(Sz)} {Sx_var} {Sy_var} {Sz_var} {Sp_var}\n")
    file.close()

    

if __name__ == "__main__":
    
    current_directory = os.getcwd()

    conf = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {'custom': {'format': '%(levelname)-8s: %(message)s'}},
    'handlers': {'to_file': {'class': 'logging.FileHandler',
                             'filename': 'log',
                             'formatter': 'custom',
                             'level': 'INFO',
                             'mode': 'a'},
                'to_stdout': {'class': 'logging.StreamHandler',
                              'formatter': 'custom',
                              'level': 'INFO',
                              'stream': 'ext://sys.stdout'}},
    'root': {'handlers': ['to_stdout', 'to_file'], 'level': 'DEBUG'},
    }
    logging.config.dictConfig(conf)

    # parser for command line arguments
    parser=argparse.ArgumentParser()
    parser.add_argument("--Lx", default='3', help="Length of cylinder")
    parser.add_argument("--Ly", default='3', help="Circumference of cylinder")
    parser.add_argument("--Jxx", default='1.0', help=" nn SxSx + SySy coupling")
    parser.add_argument("--Jz", default='1.0', help=" nn SzSz coupling")
    parser.add_argument("--hz", default='0.0', help="Magnetic field")
    parser.add_argument("--chi", default='100', help="Bond dimension")
    parser.add_argument("--init_state", default='up', help="Initial state")
    parser.add_argument("--RM", default=None, help="path for saving data")
    parser.add_argument("--max_sweep", default='50', help="Maximum number of sweeps")
    parser.add_argument("--bc_MPS", default='finite', help="'finite' or 'infinite' DMRG")
    parser.add_argument("--path", default=current_directory, help="path for saving data")
    args=parser.parse_args()

    # parameters
    Lx = int(args.Lx)
    Ly = int(args.Ly)
    Jxx = float(args.Jxx)
    Jz = float(args.Jz)
    hz = float(args.hz)
    chi = int(args.chi)
    init_state = args.init_state
    RM = args.RM
    max_sweep = int(args.max_sweep)
    bc_MPS = args.bc_MPS
    path = args.path

    if bc_MPS == 'infinite':
        bc = 'periodic'
    else:
        bc = ['open','periodic']

    # model parameters    
    model_params = {
        "Lx": Lx,
        "Ly": Ly,
        "Jxx": Jxx,
        "Jz": Jz,
        "hz": hz,
        "bc_MPS": bc_MPS,
        "bc": bc
    }

    TLAF_model = model.TLAF(model_params)

    # initial state
    if init_state == '+Sx':
        
        product_state = []
        local_state = np.array( [1., 1.] )
        for i in range( 0, Lx*Ly):
            product_state.append(local_state)

    elif init_state == 'uud':
        
        product_state = []
        for x in range(Lx):
            for y in range(Ly):
                # Shift pattern for each row (3-row periodicity)
                phase_shift = x % 3
                idx = (y - phase_shift) % 3
                if idx in [0, 1]:  # U U
                    product_state.append("up")
                else:              # D
                    product_state.append("down")

    else:
        product_state = [init_state] * (Lx * Ly)
        
    psi = MPS.from_product_state(TLAF_model.lat.mps_sites(), product_state, bc=TLAF_model.lat.bc_MPS)

    if RM == 'random':
        TEBD_params = {'N_steps': 20, 'trunc_params':{'chi_max': 100}, 'verbose': 0}
        eng = tebd.RandomUnitaryEvolution(psi, TEBD_params)
        eng.run()
        psi.canonical_form() 

    dmrg_params = {
    'mixer': True,  # setting this to True helps to escape local minima
    'trunc_params': {
        'chi_max': chi,
        'svd_min': 1.e-10
    },
    'chi_list': { 0: 16, 5: 32, 10: 64, 15: chi },
    'max_E_err': 1.0e-10,
    # 'max_S_err': 1.0e-9,
    'max_sweeps': max_sweep,
    'combine' : True
    }

    # ground state
    eng = dmrg.TwoSiteDMRGEngine(psi, TLAF_model, dmrg_params)
    E, psi = eng.run() 
    psi.canonical_form() 
    
    EE, Sx, Sy, Sz, Sx_var, Sy_var, Sz_var, Sp_var = measurements(psi)
    write_data( psi, E, EE, Sx, Sy, Sz, Sx_var, Sy_var, Sz_var, Sp_var, Lx, Ly, Jxx, hz, path )