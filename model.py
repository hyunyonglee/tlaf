# Copyright 2024 Hyun-Yong Lee

from tenpy.models.lattice import Triangular
from tenpy.models.model import CouplingModel, MPOModel
from tenpy.tools.params import Config
from tenpy.networks.site import SpinHalfSite
import numpy as np
__all__ = ['TLAF']


class TLAF(CouplingModel,MPOModel):
    
    def __init__(self, model_params):
        
        # 0) read out/set default parameters 
        if not isinstance(model_params, Config):
            model_params = Config(model_params, "TLAF")
        Lx = model_params.get('Lx', 1)
        Ly = model_params.get('Ly', 2)
        Jxx = model_params.get('Jxx', 1.)
        Jz = model_params.get('Jz', 1.)
        hz = model_params.get('hz', 0.)
        bc_MPS = model_params.get('bc_MPS', 'finite')
        bc = model_params.get('bc', 'periodic')
        
        site = SpinHalfSite( conserve=None )
        lat = Triangular( Lx=Lx, Ly=Ly, site=site, bc=bc, bc_MPS=bc_MPS )
        CouplingModel.__init__(self, lat)

        # Magnetic field
        self.add_onsite( -hz, 0, 'Sz')

        # NN XXZ
        for u1, u2, dx in self.lat.pairs['nearest_neighbors']:
            self.add_coupling(Jxx, u1, 'Sx', u2, 'Sx', dx)
            self.add_coupling(Jxx, u1, 'Sy', u2, 'Sy', dx)
            self.add_coupling(Jz, u1, 'Sz', u2, 'Sz', dx)
        
        MPOModel.__init__(self, lat, self.calc_H_MPO())