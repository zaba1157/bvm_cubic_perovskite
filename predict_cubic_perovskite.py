#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 11 16:17:05 2023

@author: zach1
"""


import os
import pandas as pd
import numpy as np
from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.symmetry.structure import SymmetrizedStructure
from scipy.optimize import minimize


module_dir = os.getcwd()
SPuDS_install_dir = os.getcwd()

class SPuDS_BVparams():
    """
    #***************************************************************
    # COPYRIGHT NOTICE
    # This table may be used and distributed without fee for
    # non-profit purposes providing 
    # 1) that this copyright notice is included and 
    # 2) no fee is charged for the table and 
    # 3) details of any changes made in this list by anyone other than
    # the copyright owner are suitably noted in the _audit_update record
    # Please consult the copyright owner regarding any other uses.
    #
    # The copyright is owned by I. David Brown, Brockhouse Institute for
    # Materials Research, McMaster University, Hamilton, Ontario Canada.
    # idbrown@mcmaster.ca
    #
    #*****************************DISCLAIMER************************
    #
    # The values reported here are taken from the literature and
    # other sources and the author does not warrant their correctness
    # nor accept any responsibility for errors.  Users are advised to
    # consult the primary sources. 
    #
    #***************************************************************
    """    
    
    def __init__(self):
        """
        Reads table of SPuDS provided bond valence parameters.
        """
        bvfile = os.path.join(module_dir, "bvparm20.cif")
        params = pd.read_csv(bvfile, sep='\s+',
                                  header=None,
                                  names=['Atom1', 'Atom1_valence',
                                         'Atom2', 'Atom2_valence',
                                         'Ro', 'B', 'ref_id', 'details'],
                                  skiprows=179,
                                  skipfooter=1,
                                  index_col=False,
                                  engine="python")
        self.params = params        
        
    def get_bv_params(self, cation, anion, cat_val, an_val):
        """
        Retrieves bond valence parameters from SPuDS table.       
        Args:
            cation str(el): cation
            anion str(el): anion
            cat_val int: cation formal oxidation state
            an_val int: anion formal oxidation state
        Returns:
            bvlist: dataframe of bond valence parameters            
        """
        bv_data = self.params
        bvlist = self.params.loc[(bv_data['Atom1'] == str(cation)) \
                                & (bv_data['Atom1_valence'] == cat_val) \
                                & (bv_data['Atom2'] == str(anion)) \
                                & (bv_data['Atom2_valence'] == an_val)]
        return bvlist.iloc[0] # First entry if multiple values exist.

class Predict_Cubic_Perovskite():
    
    def __init__(self,A1,A2,B1,B2,X1,nA1,nA2,nB1,nB2,nX1):
    
        self.bv = SPuDS_BVparams()
        self.r_cut = 6.0 # Ang
        # A-site elements
        self.A1 = A1
        self.A2 = A2
        # B-site elements
        self.B1 = B1
        self.B2 = B2
        # X-site element
        self.X1 = X1
        # Oxidation states
        self.nA1 = nA1
        self.nA2 = nA2
        self.nB1 = nB1
        self.nB2 = nB2
        self.nX1 = nX1
        # dict of {element:oxidation_state}
        self.oxidict = {A1:nA1,A2:nA2,B1:nB1,B2:nB2,X1:nX1}
        # Single or double perovksite structure
        if A1 == A2 and B1 == B2:
            self.kind = 'single'
        else:
            self.kind = 'double'
        
        self.singles = ['K','P','U','V','W','Y'] # SPuDS bv params nomenclature
        
        # Precheck sites in tabulated bond valence parameters
        for el,oxi in self.oxidict.items():
            if el in self.singles:
                el+='_'
            if el == self.X1: continue
            try:
                self.bv.get_bv_params(el,self.X1,oxi,self.nX1)
            except:
                raise Exception('Site '+el+'('+str(oxi)+')-'+
                                self.X1+'('+str(self.nX1)+')'+
                                ' not in tabulated bv parameters')            

    def Predict_lattice(self):
        
        def est_lattice(self):
            """Estimate lattice constant from optimal BVM BX6 octahedra """
            
            def calc_site_discrep(d_MX,coord_num, nM, params):
                bvs = coord_num*np.exp((float(params['Ro']) - d_MX)/float(params['B']))
                dev = abs(nM-bvs)
                return dev
            
            def optimize_BX6(self,M,nM,X,nX):        
                #BX6
                coord_num = 6.0
                if M in self.singles:
                    M += '_'                
                params = self.bv.get_bv_params(M,X,nM,nX)
                Ro = float(params['Ro'])
                result = minimize(calc_site_discrep, Ro, args=(coord_num,nM,params))
                min_bvs_BX_distance = result.x[0]
                
                return min_bvs_BX_distance
            
            if self.kind == 'single':
                lat_est = 2*optimize_BX6(self,self.B1,self.nB1,self.X1,self.nX1)
                
            elif self.kind == 'double':
                b1 = optimize_BX6(self,self.B1,self.nB1,self.X1,self.nX1)
                b2 = optimize_BX6(self,self.B2,self.nB2,self.X1,self.nX1)
                lat_est = b1+b2
                
            return lat_est        
        
        
        
        def Cubic_ABX3_structure(self,lattice_constant):
            """Sinlge ABX3 primitive cubic perovskite structure """
            els = [self.A1, self.B1, self.X1, self.X1, self.X1]
            frac_coords = [[0.5,0.5,0.5],[0,0,0],[0,0,0.5],[0,0.5,0],[0.5,0,0]]
            lattice = Lattice.cubic(lattice_constant)
            structure = Structure(lattice=lattice, species=els, coords=frac_coords,
                                  coords_are_cartesian=False)
            structure.add_oxidation_state_by_element(self.oxidict)
            return structure
        
        def Cubic_AABBX6_structure(self,lattice_constant):
            """Double AA'BB'X6 primitive cubic perovskite structure """
            els = [self.A1, self.A2, self.B1, self.B2, self.X1,
                   self.X1, self.X1, self.X1, self.X1, self.X1]
            frac_coords =[[0.25,0.25,0.25],[0.75,0.75,0.75],
                          [0,0,0],[0.5,0.5,0.5],
                          [0.75,0.75,0.25], [0.25,0.25,0.75],
                          [0.75,0.25,0.75], [0.25,0.75,0.25],
                          [0.25,0.75,0.75], [0.75,0.25,0.25]]

            lattice = Lattice.from_parameters(lattice_constant, lattice_constant,
                                              lattice_constant, 60, 60, 60)
            structure = Structure(lattice=lattice, species=els, coords=frac_coords,
                                  coords_are_cartesian=False)
            structure.add_oxidation_state_by_element(self.oxidict) 
            return structure
        
        def get_equiv_sites(s, site):
            """Find identical sites from analyzing space group symmetry."""
            sga = SpacegroupAnalyzer(s, symprec=0.01)
            sg = sga.get_space_group_operations
            sym_data = sga.get_symmetry_dataset()
            equiv_atoms = sym_data["equivalent_atoms"]
            wyckoffs = sym_data["wyckoffs"]
            sym_struct = SymmetrizedStructure(s, sg, equiv_atoms, wyckoffs)
            equivs = sym_struct.find_equivalent_sites(site)
            return equivs        

        def calc_bv_sum(self, site_val, site_el, neighbor_list):
            """Computes bond valence sum for site.
            Args:
                site_val (Integer): valence of site
                site_el (String): element name
                neighbor_list (List): List of neighboring sites and their distances
            """
            bvs = 0
            for neighbor_info in neighbor_list:
                neighbor = neighbor_info[0]
                dist = neighbor_info[1]
                neighbor_val = neighbor.species.elements[0].oxi_state
                neighbor_el = str(
                        neighbor.species.element_composition.elements[0])
                if neighbor_el in self.singles:
                    neighbor_el += '_'
                
                if np.sign(site_val) == 1 and np.sign(neighbor_val) == -1:
                    params = self.bv.get_bv_params(site_el,
                                               neighbor_el,
                                               site_val,
                                               neighbor_val)

                    bvs += np.exp((float(params['Ro']) - dist)/float(params['B']))

                elif np.sign(site_val) == -1 and np.sign(neighbor_val) == 1:
                    params = self.bv.get_bv_params(neighbor_el,
                                               site_el,
                                               neighbor_val,
                                               site_val)
                    #params['Ro'] = float(self.R0_scale)*float(params['Ro'])
                    bvs -= np.exp((float(params['Ro']) - dist)/float(params['B']))                        

            return bvs        
            
        def rcut_Cubic_GII(lattice_constant,self):
            """Computes GII for provided lattice constant.
            Args:
                lattice_constant (float): cubic primative cell lattice constant
            """            
            if self.kind == 'single':
                struct = Cubic_ABX3_structure(self,lattice_constant)
            elif self.kind == 'double':
                struct = Cubic_AABBX6_structure(self,lattice_constant)

            bond_valence_sums = []
            cutoff = self.r_cut
            pairs = struct.get_all_neighbors(r=cutoff)
            site_val_sums = {} # Cache bond valence deviations
    
            for i, neighbor_list in enumerate(pairs):
                site = struct[i]
                equivs = get_equiv_sites(struct, site)
                flag = False
    
                # If symm. identical site has cached bond valence sum difference,
                # use it to avoid unnecessary calculations
                for item in equivs:
                    if item in site_val_sums:
                        bond_valence_sums.append(site_val_sums[item])
                        site_val_sums[site] = site_val_sums[item]
                        flag = True
                        break
                if flag:
                    continue
                site_val = site.species.elements[0].oxi_state
                site_el = str(site.species.element_composition.elements[0])
                if site_el in self.singles:
                        site_el += '_'
                bvs = calc_bv_sum(self, site_val, site_el, neighbor_list)
    
                site_val_sums[site] = bvs - site_val
            gii = np.linalg.norm(list(site_val_sums.values())) /\
                  np.sqrt(len(site_val_sums))
            return gii            
            
            
        def optimize_rcut_Cubic_lattice(self):
            result = minimize(rcut_Cubic_GII, self.a_est, args=(self))
            self.a_opt = result.x[0] 
            
            
        
        self.a_est = est_lattice(self)
        
        optimize_rcut_Cubic_lattice(self)
        
        if self.kind == 'single':
            structure = Cubic_ABX3_structure(self,self.a_opt)
        elif self.kind == 'double':
            structure = Cubic_AABBX6_structure(self,self.a_opt)
        
        return {'structure':structure,'lattice_constant':self.a_opt}

if __name__ == '__main__':
    
    A1 = 'Cs'
    A2 = 'Cs'
    B1 = 'Mn'
    B2 = 'Pb'
    X1 = 'I'
    nA1 = 1
    nA2 = 1
    nB1 = 2
    nB2 = 2
    nX1 = -1
    
    Cubic = Predict_Cubic_Perovskite(A1,A2,B1,B2,X1,nA1,nA2,nB1,nB2,nX1)
    
    results_dict = Cubic.Predict_lattice()
    print(results_dict['lattice_constant'],results_dict['structure'])    
    
