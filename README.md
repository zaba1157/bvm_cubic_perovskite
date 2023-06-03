# bvm_cubic_perovskite
  The ```Predict_Cubic_Perovskite()``` class utilizes the bond valance method to provide an estimate of the rock-salt ordered primitive cubic structure for ABX<sub>3</sub> and AA'BB'X<sub>6</sub> perovskites.
  
# Requirements
  - [Pymatgen](https://pymatgen.org/)
  
# Usage
  The A-, B- and X-site elements and oxidation states are required to predict the cubic lattice constant which is obtained by minimizing the Global Instability Index (GII) with a default distance cutoff of 6 angstroms. 
  Currently, only a single X-site is supported.  
  
### Example Usage 
  ```python
  from predict_cubic_perovskite import Predict_Cubic_Perovskite

      # A-site elements
      A1 = 'Cs'
      A2 = 'Cs'

      # B-site elements
      B1 = 'Mn'
      B2 = 'Pb'

      # X-site element
      X1 = 'I'

      # Oxidation states
      nA1 = 1
      nA2 = 1
      nB1 = 2
      nB2 = 2
      nX1 = -1

      Cubic = Predict_Cubic_Perovskite(A1,A2,B1,B2,X1,nA1,nA2,nB1,nB2,nX1)
      results_dict = Cubic.Predict_lattice()

      print('Lattice Constant (Ang)':results_dict['lattice_constant'],
      'Pymatgen Structure Object': results_dict['structure'])
  ```
