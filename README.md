# rhythmite
A python, finite-difference implementation of the diagenetic reactive-transport model from the L'Heureux (2018) paper "Diagenetic Self-Organization and Stochastic Resonance in a Model of Limestone-Marl Sequences" https://doi.org/10.1155/2018/4968315.

This model was created as part of an attempt to reproduce the results of the L'Heureux paper, the original Fortran code used in the paper can be found at: https://github.com/astro-turing/Diagenetic_model_LHeureux_2018.  A different Python implementation (https://github.com/astro-turing/Integrating-diagenetic-equations-using-Python) and a Matlab code (https://github.com/MindTheGap-ERC/LMA-Matlab) were also developed as part of this project, with each applying different numerical methods in order to understand how the results of L'Heureux (2018) can be replicated.  A comparison of all the codes can be found at https://github.com/MindTheGap-ERC/Cross-comparison.

As with the original Fortran model, this code uses an upwind-scheme for the spatial derivatives in the solid phase advection equations, but instead uses a centered scheme (with optional Fiadeiro-Veronis, to match the Fortran) for the solute and porosity equations.  The time evolution uses the Scipy `solve_ivp` function, rather than an implicit Crank-Nicholson scheme.  Further details of the numerical methods and options can be found in the documentation.

# Requirements
The code requires python version >=3.9 and the following python packages, installable through `conda` or `pip` (except where specified):
- `numpy`
- `scipy`
- `matplotlib`
- `pandas`
- `numba`
- `numba-progress` (install with `pip-install numba-progress`)

# Authors
__Charlotte Summers__  
Utrecht University  
email: c.summers [at] uu.nl  
Web page: [www.uu.nl/staff/CSummers](https://www.uu.nl/staff/CSummers)  

__Cedric Thieulot__  
Utrecht University  
email: c.thieulot [at] uu.nl  
Web page: [www.cedricthieulot.net](http://www.cedricthieulot.net/)  
