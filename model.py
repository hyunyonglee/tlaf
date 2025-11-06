# Copyright 2024 Hyun-Yong Lee

from tenpy.models.lattice import Triangular
from tenpy.models.model import CouplingModel, MPOModel
from tenpy.tools.params import Config
from tenpy.networks.site import SpinSite
import numpy as np
__all__ = ['TLAF']


class TLAF(CouplingModel,MPOModel):
    
    def __init__(self, model_params):
        
        # 0) read out/set default parameters 
        if not isinstance(model_params, Config):
            model_params = Config(model_params, "TLAF")
        S = model_params.get('S', 0.5)
        Lx = model_params.get('Lx', 1)
        Ly = model_params.get('Ly', 2)
        Jxx = model_params.get('Jxx', 1.)
        Jz = model_params.get('Jz', 1.)
        G = model_params.get('G', 0.)
        PD = model_params.get('PD', 0.)
        hz = model_params.get('hz', 0.)
        bc_MPS = model_params.get('bc_MPS', 'finite')
        bc = model_params.get('bc', 'periodic')
        
        site = SpinSite( S=S, conserve=None )
        lat = Triangular( Lx=Lx, Ly=Ly, site=site, bc=bc, bc_MPS=bc_MPS )
        CouplingModel.__init__(self, lat)

        # Magnetic field
        self.add_onsite( -hz, 0, 'Sz')

        # NN XXZ
        for u1, u2, dx in self.lat.pairs['nearest_neighbors']:
            # print(f"Adding coupling between {u1} and {u2} with dx={dx}")
        
            self.add_coupling(Jxx, u1, 'Sx', u2, 'Sx', dx)
            self.add_coupling(Jxx, u1, 'Sy', u2, 'Sy', dx)
            self.add_coupling(Jz, u1, 'Sz', u2, 'Sz', dx)
        
            # Gamma terms
            if np.all(dx == np.array([1, 0])):
                self.add_coupling( G, u1, 'Sy', u2, 'Sz', dx)
                self.add_coupling( G, u1, 'Sz', u2, 'Sy', dx)

                self.add_coupling( 2*PD, u1, 'Sx', u2, 'Sx', dx)
                self.add_coupling( -2*PD, u1, 'Sy', u2, 'Sy', dx)

            elif np.all(dx == np.array([-1, 1])):
                self.add_coupling( G * np.cos(2*np.pi/3.), u1, 'Sy', u2, 'Sz', dx)
                self.add_coupling( G * np.cos(2*np.pi/3.), u1, 'Sz', u2, 'Sy', dx)
                self.add_coupling( -G * np.sin(2*np.pi/3.), u1, 'Sx', u2, 'Sz', dx)
                self.add_coupling( -G * np.sin(2*np.pi/3.), u1, 'Sz', u2, 'Sx', dx)

                self.add_coupling( 2 * PD * np.cos(2*np.pi/3.), u1, 'Sx', u2, 'Sx', dx)
                self.add_coupling( -2 * PD * np.cos(2*np.pi/3.), u1, 'Sy', u2, 'Sy', dx)
                self.add_coupling( -2 * PD * np.sin(2*np.pi/3.), u1, 'Sx', u2, 'Sy', dx)
                self.add_coupling( -2 * PD * np.sin(2*np.pi/3.), u1, 'Sy', u2, 'Sx', dx)

            else:
                self.add_coupling( G * np.cos(2*np.pi/3.), u1, 'Sy', u2, 'Sz', dx)
                self.add_coupling( G * np.cos(2*np.pi/3.), u1, 'Sz', u2, 'Sy', dx)
                self.add_coupling( G * np.sin(2*np.pi/3.), u1, 'Sx', u2, 'Sz', dx)
                self.add_coupling( G * np.sin(2*np.pi/3.), u1, 'Sz', u2, 'Sx', dx)

                self.add_coupling( 2 * PD * np.cos(2*np.pi/3.), u1, 'Sx', u2, 'Sx', dx)
                self.add_coupling( -2 * PD * np.cos(2*np.pi/3.), u1, 'Sy', u2, 'Sy', dx)
                self.add_coupling( 2 * PD * np.sin(2*np.pi/3.), u1, 'Sx', u2, 'Sy', dx)
                self.add_coupling( 2 * PD * np.sin(2*np.pi/3.), u1, 'Sy', u2, 'Sx', dx)
            
        MPOModel.__init__(self, lat, self.calc_H_MPO())