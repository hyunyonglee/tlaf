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



def measurements(psi, Lx, Ly):
    # Entanglement entropy
    EE = psi.entanglement_entropy()

    # Local observables
    Sx = psi.expectation_value("Sx")
    Sy = psi.expectation_value("Sy")
    Sz = psi.expectation_value("Sz")
    
    Sx = np.real(Sx)
    Sy = np.real(Sy)
    Sz = np.real(Sz)
    
    # Scalar spin chirality
    chis = []
    for x in range(Lx):
        for y in range(Ly):
            
            i = x * Ly + y
            j = (x+1) * Ly + y
            if y == Ly-1:
                k1 = x * Ly
            else:
                k1 = i + 1

            if y == 0:
                k2 = j + Ly - 1
            else:
                k2 = j - 1

            # Expectation values for the scalar spin chirality
            term_x1 = psi.expectation_value_term([("Sx", i), ("Sy", j), ("Sz", k1)])
            term_x1 = term_x1 - psi.expectation_value_term([("Sx", i), ("Sz", j), ("Sy", k1)])
            term_y1 = psi.expectation_value_term([("Sy", i), ("Sz", j), ("Sx", k1)])
            term_y1 = term_y1 - psi.expectation_value_term([("Sy", i), ("Sx", j), ("Sz", k1)])
            term_z1 = psi.expectation_value_term([("Sz", i), ("Sx", j), ("Sy", k1)])
            term_z1 = term_z1 - psi.expectation_value_term([("Sz", i), ("Sy", j), ("Sx", k1)])
            chi1 = term_x1 + term_y1 + term_z1

            term_x2 = psi.expectation_value_term([("Sx", i), ("Sy", j), ("Sz", k2)])
            term_x2 = term_x2 - psi.expectation_value_term([("Sx", i), ("Sz", j), ("Sy", k2)])
            term_y2 = psi.expectation_value_term([("Sy", i), ("Sz", j), ("Sx", k2)])
            term_y2 = term_y2 - psi.expectation_value_term([("Sy", i), ("Sx", j), ("Sz", k2)])
            term_z2 = psi.expectation_value_term([("Sz", i), ("Sx", j), ("Sy", k2)])
            term_z2 = term_z2 - psi.expectation_value_term([("Sz", i), ("Sy", j), ("Sx", k2)])
            chi2 = term_x2 + term_y2 + term_z2

            chis.append(np.real_if_close(chi1))
            chis.append(np.real_if_close(-chi2)) # (-) sign
            
    return EE, Sx, Sy, Sz, chis


def write_data( psi, E, EE, Sx, Sy, Sz, chis, Lx, Ly, Jxx, G, PD, hz, path, wavefunc=False ):

    ensure_dir(path+"/observables/")
    ensure_dir(path+"/mps/")

    if wavefunc:
        data = {"psi": psi}
        with h5py.File(path+"/mps/psi_Lx_%d_Ly_%d_Jxx_%.2f_G_%.2f_PD_%.2f_hz_%.2f.h5" % (Lx, Ly, Jxx, G, PD, hz), 'w') as f:
            hdf5_io.save_to_hdf5(f, data)

    file_EE = open(path+"/observables/EE.txt","a", 1)    
    file_Sx = open(path+"/observables/Sx.txt","a", 1)
    file_Sy = open(path+"/observables/Sy.txt","a", 1)
    file_Sz = open(path+"/observables/Sz.txt","a", 1)
    file_chis = open(path+"/observables/chis.txt","a", 1)
        
    file_EE.write(f"{Jxx} {G} {PD} {hz}  {'  '.join(map(str, EE))}\n")
    file_Sx.write(f"{Jxx} {G} {PD} {hz}  {'  '.join(map(str, Sx))}\n")
    file_Sy.write(f"{Jxx} {G} {PD} {hz}  {'  '.join(map(str, Sy))}\n")
    file_Sz.write(f"{Jxx} {G} {PD} {hz}  {'  '.join(map(str, Sz))}\n")
    file_chis.write(f"{Jxx} {G} {PD} {hz}  {'  '.join(map(str, chis))}\n")
    
    file_EE.close()
    file_Sx.close()
    file_Sy.close()
    file_Sz.close()
    file_chis.close()
    
    #
    file = open(path+"/observables.txt","a", 1)    
    file.write(f"{Jxx} {G} {PD} {hz} {E} {np.max(EE)} {np.mean(Sx)} {np.mean(Sy)} {np.mean(Sz)} {np.mean(chis)} \n")
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
    parser.add_argument("--Spin", default='0.5', help="Spin value")
    parser.add_argument("--Lx", default='3', help="Length of cylinder")
    parser.add_argument("--Ly", default='3', help="Circumference of cylinder")
    parser.add_argument("--Jxx", default='1.0', help=" nn SxSx + SySy coupling")
    parser.add_argument("--Jz", default='1.0', help=" nn SzSz coupling")
    parser.add_argument("--G", default='0.0', help=" nn Gamma coupling")
    parser.add_argument("--PD", default='0.0', help=" nn Gamma coupling")
    parser.add_argument("--hz", default='0.0', help="Magnetic field")
    parser.add_argument("--chi", default='100', help="Bond dimension")
    parser.add_argument("--init_state", default='up', help="Initial state")
    parser.add_argument("--RM", default=None, help="path for saving data")
    parser.add_argument("--max_sweep", default='50', help="Maximum number of sweeps")
    parser.add_argument("--bc_MPS", default='infinite', help="'finite' or 'infinite' DMRG")
    parser.add_argument("--path", default=current_directory, help="path for saving data")
    parser.add_argument("--wavefunc", action='store_true', help="Save wavefunction")
    parser.add_argument("--load", action='store_true', help="Load wavefunction from file")
    args=parser.parse_args()

    # parameters
    S = float(args.Spin)
    Lx = int(args.Lx)
    Ly = int(args.Ly)
    Jxx = float(args.Jxx)
    Jz = float(args.Jz)
    G = float(args.G)
    PD = float(args.PD)
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
        "S": S,
        "Lx": Lx,
        "Ly": Ly,
        "Jxx": Jxx,
        "Jz": Jz,
        "G": G,
        "PD": PD,
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

    elif init_state == '+Sy':
        product_state = []
        local_state = np.array([1, 1+1j], dtype=complex)
        for i in range( 0, Lx*Ly):
            product_state.append(local_state)

    elif init_state == '+Sy1':
        product_state = []
        local_state = np.array([1j, 0, 1j], dtype=complex)
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

    elif init_state == 'Y':
        
        product_state = []
        for x in range(Lx):
            for y in range(Ly):
                # Shift pattern for each row (3-row periodicity)
                phase_shift = x % 3
                idx = (y - phase_shift) % 3
                if idx == 0:  # U U
                    # product_state.append(np.array([1+0.7, 1]))
                    product_state.append(np.array([1+0.5, +0.5], dtype=complex))
                elif idx == 1:  # U U
                    product_state.append(np.array([1+0.5, -0.5], dtype=complex))
                else:              # D
                    product_state.append(np.array([0, 1], dtype=complex))

    else:
        product_state = [init_state] * (Lx * Ly)
        
    psi = MPS.from_product_state(TLAF_model.lat.mps_sites(), product_state, dtype=complex, bc=TLAF_model.lat.bc_MPS)
    psi.canonical_form()

    if args.load:

        print("Loading wavefunction from file...")
        if S==0.5:
            file_path = "/home/hylee/tlaf/reference1.h5"
        elif S==1.0:
            file_path = "/home/hylee/tlaf/reference2.h5"
        
        with h5py.File(file_path, 'r') as f:
            data = hdf5_io.load_from_hdf5(f)
            psi = data["psi"]
        print("Wavefunction loaded.")

    # chi list
    if args.load:
        chi_list = {0: 16, 20: chi}
    else:
        chi_list = {0: 16, 5: 32, 10: 64, 15: 128, 20: chi}

    if RM == 'random':
        TEBD_params = {'N_steps': 20, 'trunc_params':{'chi_max': 32}, 'verbose': 0}
        eng = tebd.RandomUnitaryEvolution(psi, TEBD_params)
        eng.run()
        psi.canonical_form() 

    dmrg_params = {
    'mixer':  dmrg.SubspaceExpansion,
    'mixer_params': {
        'amplitude': 1.e-3,
        'decay': 2.0,
        'disable_after': 20
    }, # setting this to True helps to escape local minima
    'trunc_params': {
        'chi_max': chi,
        'svd_min': 1.e-8 # 1.e-10
    },
    'chi_list': chi_list,
    'max_E_err': 1.0e-8, # 1.0e-10,
    # 'max_S_err': 1.0e-9,
    'max_sweeps': max_sweep,
    'combine' : True
    }

    # ground state
    eng = dmrg.TwoSiteDMRGEngine(psi, TLAF_model, dmrg_params)
    E, psi = eng.run() 
    psi.canonical_form() 
    
    EE, Sx, Sy, Sz, chis = measurements(psi, Lx, Ly)
    write_data( psi, E, EE, Sx, Sy, Sz, chis, Lx, Ly, Jxx, G, PD, hz, path, wavefunc=args.wavefunc )