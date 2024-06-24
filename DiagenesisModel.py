
#!/usr/bin/env python3

###  L'Heureux Diagenesis modelling ###

import numpy as np
from scipy.integrate import solve_ivp
from numba import jit, float64, int64, boolean
from numba.experimental import jitclass
import time

from saveOutput import export_to_ascii, export_to_vtu, export_to_vtu2

## model for testing different solvers on the L'Heureux (2018) equations 

###############################################################################
@jit(nopython=True)
def heaviside(x,xbot,xtop,xscale, smooth_switch):
    
    if (smooth_switch==False):
        if x < xbot/xscale and x > xtop/xscale:
          #print (x,1)
          return 1.0
        else:
          #print (x,0)
          return 0.0
    else:
        val=0.5*(1+np.tanh((x-xtop/xscale)*500)) *0.5*(1+np.tanh((xbot/xscale-x)*500))
        return val

###############################################################################

# for the jit-compiling, we need to specify the type of all parameters in the class
spec = [
    ('g', float64),
    ('K_C', float64),
    ('K_A', float64),
    ('ADZ_bot', float64),
    ('ADZ_top', float64),
    ('m', float64),
    ('n', float64),
    ('sed_rate', float64),
    ('rho_w', float64),
    ('D_Ca_0', float64),
    ('D_CO', float64),
    ('b', float64),
    ('beta', float64),
    ('k1', float64),
    ('k2', float64),
    ('k3', float64),
    ('k4', float64),
    ('muA', float64),
    ('lamb', float64),
    ('nu1', float64),
    ('nu2', float64),
    ('AR_0', float64),
    ('CA_0', float64),
    ('c_ca_0', float64),
    ('c_co_0', float64),
    ('phi_0', float64),
    ('rho_s0', float64),
    ('AR_init', float64),
    ('CA_init', float64),
    ('c_ca_init', float64),
    ('c_co_init', float64),
    ('phi_init', float64),
    ('delta', float64),
    ('phi_NR', float64),
    ('phi_inf', float64),
    ('d_phi', float64),
    ('x_scale', float64),
    ('t_scale', float64),
    ('v_scale', float64),
    ('Da', float64),
    ('upwind_switch', int64),
    ('smooth_switch', boolean),
    ('FV_switch', int64)
]

# This class contains the functions and parameters that are used to calculate
# the RHS of eqns 40-43 of L'Heureux (2018)
@jitclass(spec)
class LHeureux:
    
    # define all params as instance vars
    # this way we can easily modify them at the instance level
    # i.e. for parameter searches
    def __init__(self):
        
        # physical constants
        self.g = 9.81*100           # gravitational acceleration (cm/s^2)
        
        # model parameters
        self.K_C = 10**-6.37        # Calcite solubility (M^2)
        self.K_A = 10**-6.19        # Aragonite solubility (M^2)
        self.ADZ_top = 50           # top of the Aragonite dissolution zone (cm)
        self.ADZ_bot = 150          # bottom of the Aragonite dissolution zone (cm)
        self.sed_rate = 0.1         # sedimentation rate (cm/a)
        self.m = 2.48               # Power in def. of Omega_A (Eqn. 45)
        self.n = 2.8                # Power in def. of Omega_C (Eqn. 45)
        self.rho_w = 1.023          # density of water (g/cm^3)
        self.D_Ca_0 = 131.9         # diffusion coefficient of Ca (cm^2/a) (used to scale to dimensionless units)
        self.D_CO = 272.6           # scaled diffusion coefficient of CO3 (cm^2/a)
        
        self.b = 5.0e-4*0.2             # sediment compressibility (Pa^-1)
        self.beta = 0.1             # Hydraulic conductivity constant (cm/a)
        
        self.k1 = 1.0               # reaction rate constants (a^-1)
        self.k2 = self.k1         
        self.k3 = 0.1
        self.k4 =self.k3
        self.muA = 100.09           # (g/mol)
        
        self.lamb = self.k3/self.k2    # constant in reaction terms  
        self.nu1 = self.k1/self.k2     # constant in reaction terms
        self.nu2 = self.k4/self.k3     # constant in reaction terms
        
        # upper boundary condition values (x=0)
        self.AR_0 = 0.6                
        self.CA_0 = 0.3
        self.c_ca_0 = 0.326e-3/np.sqrt(self.K_C)
        self.c_co_0 = 0.326e-3/np.sqrt(self.K_C)
        self.phi_0 = 0.8            # steady state 0.6, oscillations 0.8

        self.rho_s0 = 2.95*self.AR_0+2.71*self.CA_0+2.8*(1-(self.AR_0+self.CA_0)) # initial sediment density (g/cm^3)

        # initial condtions
        self.AR_init = self.AR_0
        self.CA_init = self.CA_0
        self.c_ca_init = self.c_ca_0
        self.c_co_init = self.c_co_0
        self.phi_init = 0.8         # steady state 0.5, oscillations 0.8

        # more params (which depend on initial/boundary conditions)
        self.delta = self.rho_s0/(self.muA*np.sqrt(self.K_C))   # part of the Ca, CO3 reaction terms (cm^-3)
        self.phi_NR = self.phi_init                             # porosity in the absence of reactions
        self.phi_inf = 0.01                                     # "a parameter" in Eqn 23

        # porosity diffusion coefficient
        self.d_phi = self.beta*(self.phi_init**3 / ( 1 - self.phi_init ) )*\
                    (1 / ( self.b*self.g*self.rho_w*( self.phi_NR - self.phi_inf ) ) )*\
                    (1 - np.exp( -10*( 1 - self.phi_init ) / self.phi_init) )*( 1 / self.D_Ca_0 )
        
        # scaling factors, used to convert to dimensionless units
        self.x_scale = self.D_Ca_0/self.sed_rate
        self.t_scale = self.D_Ca_0/self.sed_rate**2
        self.v_scale = self.sed_rate
        
        self.Da = self.k2*self.t_scale    # Damkohler number
        
        self.upwind_switch = 1          # optionally use upwind derivatives in AR, CA eqns
        self.smooth_switch = False      # option to use smoothed Heaviside fn for ADZ
        self.FV_switch = 0              # optionally use the Fiadeiro-Veronis scheme ofr c_ca, co, phi
    
    ################################################
    # hydraulic conductivity
    ################################################

    def K(self,phi):
        return self.beta*(phi**3/(1-phi)**2)*(1-np.exp(-10*(1-phi)/phi))
    
    ################################################
    # velocities
    ################################################

    def U(self, phi):
        # solid velocity, eqn 46
        u = 1-(1/self.sed_rate)*\
            (self.K(self.phi_0) * ( 1 - self.phi_0 ) - self.K(phi)*(1-phi))*\
            (self.rho_s0 / self.rho_w - 1 )
        return u

    def W(self,phi):
        # solute velocity, eqn 47
        w = 1-(1/self.sed_rate)*\
            (self.K(self.phi_0) * ( 1 - self.phi_0 ) + self.K(phi)*(1-phi)**2/phi)*\
            (self.rho_s0 / self.rho_w - 1 )
        return w
    
    ################################################
    # Saturation factors 
    ################################################

    def Omega_A(self, c_ca, c_co, x):
        
        sp = c_ca*c_co*self.K_C/self.K_A - 1 
        Omega_PA = (max(0.0,sp))**self.m 
        
        sa = 1 - c_ca*c_co*self.K_C/self.K_A
        Omega_DA = (max(0.0,sa))**self.m * heaviside(x,self.ADZ_bot,self.ADZ_top,self.x_scale, self.smooth_switch)
        
        return Omega_DA - self.nu1*Omega_PA

    def Omega_C(self, c_ca, c_co): 
        
        sp = c_ca*c_co - 1
        Omega_PC = (max(0.0,sp))**self.n
        
        sa = 1 - c_ca*c_co 
        Omega_DC = (max(0.0, sa))**self.n
        
        return Omega_PC - self.nu2*Omega_DC 
    
    ################################################
    # full reaction terms
    ################################################
    
    # Aragonite
    def R_AR(self, AR, CA, c_ca, c_co, x):
        return - self.Da*( ( 1 - AR ) * AR * self.Omega_A(c_ca, c_co, x) +\
                        self.lamb * AR * CA * self.Omega_C(c_ca, c_co) )
    
    #Calcite
    def R_CA(self, AR, CA, c_ca, c_co, x):
        return self.Da*( self.lamb * ( 1 - CA ) * CA * self.Omega_C(c_ca, c_co) +\
                        AR * CA * self.Omega_A(c_ca, c_co, x) )
    
    # Ca ions
    def R_c_ca(self, AR, CA, c_ca, c_co, phi, x):
        return self.Da*( ( 1 - phi ) / phi ) * (self.delta - c_ca)*\
               ( AR * self.Omega_A(c_ca, c_co, x) - self.lamb * CA * self.Omega_C(c_ca, c_co) )
    
    # CO3 ions
    def R_c_co(self, AR, CA, c_ca, c_co, phi, x):
        return self.Da*( ( 1 - phi ) / phi ) * (self.delta - c_co)*\
               ( AR * self.Omega_A(c_ca, c_co, x) - self.lamb * CA * self.Omega_C(c_ca, c_co) )
    
    # porosity
    def R_phi(self, AR, CA, c_ca, c_co, phi, x):
        return self.Da*( 1 - phi )*\
               ( AR * self.Omega_A(c_ca, c_co, x) - self.lamb * CA * self.Omega_C(c_ca, c_co) )
    
    ################################################
    # diffusion coefficients, dissolved ions (eqn 6)
    ################################################
    
    # Ca ions
    def d_c_ca(self, phi):
        # scaled with D_Ca_0
        return 1.0/( 1 - 2*np.log(phi) )
    
    #CO3 ions
    def d_c_co(self, phi):
        # scaled with D_Ca_0
        return self.D_CO/self.D_Ca_0*(1 / ( 1 - 2*np.log(phi) ) )

    # calculate the grid Peclet no.
    # use already calculated velocity to reduce fn calls
    def Pec_phi(self, w, h):
        return w*h/(2*self.d_phi) 
    
    def Pec_c_ca(self,w, h, phi):
        return w*h/(2*self.d_c_ca(phi))
    
    def Pec_c_co(self, w, h, phi):
        return w*h/(2*self.d_c_co(phi))
    
    # also calculate the sigmas for the FV scheme
    def sigma_ca(self, w, h, phi):
        pec = self.Pec_c_ca(w,h,phi)
        return 1/np.tanh(pec) - 1/pec

    def sigma_co(self, w, h, phi):
        pec = self.Pec_c_co(w,h,phi)
        return 1/np.tanh(pec) - 1/pec
    
    def sigma_phi(self, w, h):
        pec = self.Pec_phi(w,h)
        return 1/np.tanh(pec) - 1/pec

    #################################################################
    ##### functions which calculate the full RHS of eqns 40-43 ######
    #####   with spatial derivative stencils applied for MOL   ######
    #################################################################

    # Aragonite (eqn 40)
    def RHS_AR(self, AR, CA, c_ca, c_co, phi, x, h):
        dAR_dt = np.zeros(len(x))
        u = self.U(phi)
        
        for i in range(0,len(x)):
            # x = 0 BC, Dirichlet
            if (i==0):
                dAR_dt[i]= 0
            # x = Lx, no prescribed BC
            elif (i==len(x)-1):
                dAR_dt[i]= -u[i]*(AR[i]-AR[i-1])/h + self.R_AR(AR[i],CA[i],c_ca[i],c_co[i],x[i])
            else:
                dAR_dt[i]= -u[i]*( (AR[i+1]-AR[i-1])/(2*h) - self.upwind_switch*np.sign(u[i])*h/2*(AR[i+1] - 2*AR[i] + AR[i-1])/h**2 )\
                    + self.R_AR(AR[i],CA[i],c_ca[i],c_co[i], x[i])    
        return dAR_dt

    # Calcite (eqn 41)
    def RHS_CA(self, AR, CA, c_ca, c_co, phi, x, h):
        dCA_dt = np.zeros(len(x))
        u = self.U(phi)
        for i in range(0,len(x)):
            # x = 0 BC, Dirichlet
            if (i==0):
                dCA_dt[i]= 0
            # x = Lx, no prescribed BC
            elif (i==len(x)-1):
                dCA_dt[i]= -u[i]*( CA[i]-CA[i-1] ) / h + self.R_CA(AR[i], CA[i], c_ca[i], c_co[i], x[i])
            else:
                dCA_dt[i]= -u[i]*( ( CA[i+1]-CA[i-1])/(2*h) - self.upwind_switch*np.sign(u[i])*h/2*(CA[i+1] - 2*CA[i] + CA[i-1])/h**2 )\
                    +self.R_CA(AR[i],CA[i],c_ca[i],c_co[i],x[i])
        return dCA_dt
    
    # Ca ions (eqn 42)
    def RHS_c_ca(self, AR, CA, c_ca, c_co, phi, x, h):
        dc_ca_dt = np.zeros(len(x))
        w = self.W(phi)
        phi_half = np.zeros(len(x)-1)
        d_ca_half = np.zeros(len(x)-1)
        
        for i in range(0,len(x)-1):
            phi_half[i] = ( phi[i+1] + phi[i] ) / 2
            d_ca_half[i] = ( self.d_c_ca(phi[i+1]) + self.d_c_ca(phi[i]) ) / 2
        
        for i in range(0,len(x)):
            # x = 0 BC, Dirichlet
            if (i==0):
                dc_ca_dt[i] = 0
            # x = Lx BC, df/dx = 0
            elif (i==len(x)-1):
                dc_ca_dt[i] =  ( 1 / phi[i] ) * ( phi[i-2] * self.d_c_ca(phi[i-2]) * c_ca[i-2] -\
                                                   phi[i-1] * self.d_c_ca(phi[i-1]) * c_ca[i-1] ) / h**2  +\
                              self.R_c_ca(AR[i], CA[i], c_ca[i], c_co[i], phi[i], x[i]) 
            else:
                dc_ca_dt[i] = - w[i]*( (c_ca[i+1] - c_ca[i-1]) / (2*h) +\
                                    -self.FV_switch*self.sigma_ca(w[i], h, phi[i])*h/2*(c_ca[i+1] - 2*c_ca[i] + c_ca[i-1])/(2*h) ) +\
                              ( 1 / phi[i] ) * ( phi_half[i]   * d_ca_half[i]  *(c_ca[i+1] - c_ca[i]) -\
                                                 phi_half[i-1] * d_ca_half[i-1]*(c_ca[i] - c_ca[i-1]) ) / h**2 +\
                              self.R_c_ca(AR[i], CA[i], c_ca[i], c_co[i], phi[i], x[i])
        return dc_ca_dt
    
    # CO3 ions (eqn 42)
    def RHS_c_co(self, AR, CA, c_ca, c_co, phi, x, h):
        dc_co_dt = np.zeros(len(x))
        w = self.W(phi)
        phi_half = np.zeros(len(x)-1)
        d_co_half = np.zeros(len(x)-1)
        
        for i in range(0,len(x)-1):
            phi_half[i] = ( phi[i+1] + phi[i] ) / 2
            d_co_half[i] = ( self.d_c_co(phi[i+1]) + self.d_c_co(phi[i]) ) / 2
        for i in range(0,len(x)):
            # x = 0 BC, Dirichlet
            if (i==0):
                dc_co_dt[i] = 0
            # x = Lx BC, df/dx = 0
            elif (i==len(x)-1):
                dc_co_dt[i] = ( 1 / phi[i] )*( phi[i-2] * self.d_c_co(phi[i-2]) * c_co[i-2] -\
                                               phi[i-1] * self.d_c_co(phi[i-1]) * c_co[i-1] ) / h**2 +\
                              self.R_c_co(AR[i], CA[i], c_ca[i], c_co[i], phi[i], x[i]) 
            else:
                dc_co_dt[i] = - w[i]*( (c_co[i+1] - c_co[i-1]) / (2*h) +\
                              -self.FV_switch*self.sigma_co(w[i], h, phi[i])*h/2*(c_co[i+1] - 2*c_co[i] + c_co[i-1])/(2*h) ) +\
                              (1 / phi[i] )*( phi_half[i]   * d_co_half[i]   * ( c_co[i+1] - c_co[i] ) -\
                                              phi_half[i-1] * d_co_half[i-1] * ( c_co[i] - c_co[i-1] ) ) / h**2 +\
                              self.R_c_co(AR[i], CA[i], c_ca[i], c_co[i], phi[i], x[i])
        return dc_co_dt

    # porosity (eqn 43)
    def RHS_phi(self, AR, CA, c_ca, c_co, phi, x, h):
        dphi_dt = np.zeros(len(x))
        w = self.W(phi)
        for i in range(0,len(x)):
            # x = 0 BC, Dirichlet
            if (i==0):
                dphi_dt[i] = 0
            # x = Lx BC, df/dx = 0
            elif (i==len(x)-1):
                dphi_dt[i] = - ( w[i] * phi[i] - w[i-1] * phi[i-1] ) / h +\
                             self.d_phi*( phi[i-2] - phi[i-1] ) / h**2 +\
                             self.R_phi(AR[i], CA[i], c_ca[i], c_co[i], phi[i], x[i])
            else:
                dphi_dt[i] = -( w[i+1] * phi[i+1] - w[i-1] * phi[i-1] ) / (2*h) +\
                    self.FV_switch*self.sigma_phi(w[i], h)*h/2*(phi[i+1]*w[i+1] - 2*phi[i]*w[i] + phi[i-1]*w[i-1])/(2*h) +\
                            self.d_phi * ( phi[i+1] - 2*phi[i] + phi[i-1] ) / h**2 +\
                            self.R_phi(AR[i], CA[i], c_ca[i], c_co[i], phi[i], x[i])
        return dphi_dt
   
    ########################################## 
    # combined call for the RHS functions, 
    # for use in the scipy ivp_solve function
    ########################################## 

    def X_RHS(self, t, X, nnx, x, h):
        
        dAR_dt   = self.RHS_AR(  X[0:nnx], X[nnx:2*nnx], X[2*nnx:3*nnx], X[3*nnx:4*nnx], X[4*nnx:5*nnx], x, h)
        dCA_dt   = self.RHS_CA(  X[0:nnx], X[nnx:2*nnx], X[2*nnx:3*nnx], X[3*nnx:4*nnx], X[4*nnx:5*nnx], x, h)
        dc_ca_dt = self.RHS_c_ca(X[0:nnx], X[nnx:2*nnx], X[2*nnx:3*nnx], X[3*nnx:4*nnx], X[4*nnx:5*nnx], x, h)
        dc_co_dt = self.RHS_c_co(X[0:nnx], X[nnx:2*nnx], X[2*nnx:3*nnx], X[3*nnx:4*nnx], X[4*nnx:5*nnx], x, h)
        dphi_dt  = self.RHS_phi( X[0:nnx], X[nnx:2*nnx], X[2*nnx:3*nnx], X[3*nnx:4*nnx], X[4*nnx:5*nnx], x, h)
        
        
        return np.concatenate((dAR_dt, dCA_dt, dc_ca_dt, dc_co_dt, dphi_dt)) 

###############################################################################

# create instance of the model class
lh = LHeureux()

# labels corresponding the the soln variables, for print statements
labels=['AR  ', 'CA  ', 'c_ca', 'c_co', 'phi ']

# run settings 
verbose = True        # print out extra info at each step, for debugging
method = 'Euler'      # choice of method for time integration
                      # options currently: 'Euler','RK23','RK45','DOP853'
restart = False       # flag to determine a restart from file
restrt_tstep = 10000     # tstep number of restart file to use


steadyCheck = False    # switch for optional steady state checking
tol = 1e-6             # tolerance for minimum variation between steps, needs tuning to tstep?
count_threshold = 100  # tsteps for which solution variation must remain under to trigger steady state
steady_count = 0       # variable to track how many continous steps a steady state was present in

print_freq = 10000         # frequency of print statements in Euler mode
output_freq = 1000         # frequency of soln storage
checkpnt_freq = 14e7    # frequency of restart file write
stats_freq = 1000         # frequecy to write to the _stats files

########################### space grid setup ##################################
nnx = 200                   # number of grid points
L_x = 500/lh.x_scale        # Physical size of domain cm/x_scale
h = L_x/(nnx-1)             # spatial step size
x = np.linspace(0, L_x,nnx) # position array


########################### initial conditions ################################
X_new = np.zeros([nnx*5])   # vector for new X calculated at each point

if (restart):
    # read the restart file
    # store in X
    rstrt = np.loadtxt('restrt_%08d.ascii'%(restrt_tstep))
    
    t0 = rstrt[0] # set initial time with the time the restart was produced
    
    X = rstrt[1:] # fill X arr with the read in values
      
else:
    # for new run
    t0  = 0.0
    
    # set initial conditions from the values defined in LHeureux class
    AR   = np.ones(nnx)*lh.AR_init
    CA   = np.ones(nnx)*lh.CA_init
    c_ca = np.ones(nnx)*lh.c_ca_init
    c_co = np.ones(nnx)*lh.c_co_init
    phi  = np.ones(nnx)*lh.phi_init

    # set the phi boundary in case it's different
    phi[0] = lh.phi_0

    # combine to create initial X
    X = np.concatenate([AR, CA, c_ca, c_co, phi])
    
    
######################## time integration values ##############################

tf = 250000/lh.t_scale # final sim time in a, scaled to dimensionless form 

# set the timestep manually, ONLY used in Euler mode
delta_t = 1e-6#0.001/lh.t_scale   # timestep in a, 1.13e-2/tsc = 10^-6 in scaled time
t_arr = np.arange(t0, tf, delta_t)

t_eval = t_arr[::output_freq] # times at which to store the solution, for ivp routines

# open stats files
velstats_file=open('stats_vel.ascii',"w")
AR_file=open('stats_AR.ascii',"w")
CA_file=open('stats_CA.ascii',"w")
cca_file=open('stats_c_ca.ascii',"w")
cco_file=open('stats_c_co.ascii',"w")
phi_file=open('stats_phi.ascii',"w")

###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################

if (method=='Euler'):
    
    # output array for whole solution
    soln = []
    vel_U = []
    vel_W = []
    
    U_temp = np.zeros(nnx)
    W_temp = np.zeros(nnx)
    
    start = time.time()
    for i in range(0,len(t_arr)):
        
        if (i%print_freq==0):
            print('*****istep=',i,'***********************')
            print('t=%.2e / tf=%.2e' %(t_arr[i]*lh.t_scale,tf*lh.t_scale))
        
        if (i%output_freq==0):
            # first append current soln to storage
            soln.append(X)
        
        # calculate the RHS
        dX_dt = lh.X_RHS(t_arr, X, nnx, x, h)
        
        # calculate the new X using forward Euler
        X_new = X + delta_t*dX_dt
        
        #######################################################################
        ##################### phi, c_ca, c_co BC at nnx #######################
        # want a flat derivative here so we set phi(nnx-1) - phi(nnx-2)
        X_new[5*nnx-1] = X_new[5*nnx-2]
        X_new[4*nnx-1] = X_new[4*nnx-2]
        X_new[3*nnx-1] = X_new[3*nnx-2]
        
        U_temp = lh.U(X_new[4*nnx:5*nnx])
        W_temp = lh.W(X_new[4*nnx:5*nnx])
        
        # check for negative concentrations
        if (i%print_freq==0):
            print('Negative AR, CA, ca, co, phi:%i, %i, %i, %i, %i'\
                  %(np.count_nonzero(X_new[0:nnx] < 0), np.count_nonzero(X_new[nnx:2*nnx] < 0),\
                    np.count_nonzero(X_new[2*nnx:3*nnx] < 0), np.count_nonzero(X_new[3*nnx:4*nnx] < 0),\
                    np.count_nonzero(X_new[4*nnx:5*nnx] < 0)) )
        
        # apply limits if variables become unphysical
        for j in range(0,5):
            
            if (j==0 or j==1):
                X_new[j*nnx:(j+1)*nnx] = np.clip(X_new[j*nnx:(j+1)*nnx], 0.0, 1.0)
            elif (j==4):
                X_new[j*nnx:(j+1)*nnx] = np.clip(X_new[j*nnx:(j+1)*nnx], 0.01, 1.0)
            else:
                X_new[j*nnx:(j+1)*nnx] = np.clip(X_new[j*nnx:(j+1)*nnx], 0.0, 1.0e5)

        if (i%stats_freq == 0):
            velstats_file.write("%d %4e %4e %4e %4e \n" % (i,np.min(U_temp),\
                                                             np.max(U_temp),\
                                                             np.min(W_temp),\
                                                             np.max(W_temp)))
            velstats_file.flush()

            AR_file.write("%d %4e %4e \n" %(i,np.min(X_new[0*nnx:1*nnx]),\
                                              np.max(X_new[0*nnx:1*nnx])))
            AR_file.flush()
        
            CA_file.write("%d %4e %4e \n" %(i,np.min(X_new[1*nnx:2*nnx]),\
                                              np.max(X_new[1*nnx:2*nnx])))
            CA_file.flush()
        
            cca_file.write("%d %4e %4e \n" %(i,np.min(X_new[2*nnx:3*nnx]),\
                                               np.max(X_new[2*nnx:3*nnx])))
            cca_file.flush()
        
            cco_file.write("%d %4e %4e \n" %(i,np.min(X_new[3*nnx:4*nnx]),\
                                               np.max(X_new[3*nnx:4*nnx])))
            cco_file.flush()
        
            phi_file.write("%d %4e %4e \n" %(i,np.min(X_new[4*nnx:5*nnx]),\
                                               np.max(X_new[4*nnx:5*nnx])))
            phi_file.flush()

            
        # print out min, max values for each t step if needed
        if verbose and i%print_freq==0:
           for j in range(0,5):
               # maximum and min soln values
               print('%s (m,M) = %.3f , %.3f' %(labels[j],np.min(X_new[j*nnx:(j+1)*nnx]),\
                                                          np.max(X_new[j*nnx:(j+1)*nnx])))
           print('U    (m,M) = %.3f , %.3f' %(np.min(U_temp),\
                                              np.max(U_temp)))
           print('W    (m,M) = %.3f , %.3f' %(np.min(W_temp),\
                                              np.max(W_temp)))
            
        if (i%output_freq==0):
            vel_U.append(U_temp)            
            vel_W.append(W_temp)            
        
        # check for nans, exit if we see them
        isnan = np.isnan(X_new)
        if (np.any(isnan)):
            print('nan encountered, exiting')
            
            # need to make sure t_eval matches soln for output
            t_eval = t_eval[:len(soln)]
            
            break
        
        # steady state detection
        
        if (steadyCheck):
            # check to see if we have a steady state
            
            
            # maybe we want this to check for a couple of timesteps in a row ???
            X_diff = np.absolute(X_new - X)
            
            if (np.max(X_diff) <= tol):
                # all values are below tolerance this step
                steady_count += 1
            else:
                # not under the threshold, reset counter
                steady_count = 0
            
            
            if (steady_count >= count_threshold):
                # we've reached conditions for steady state
                # exit
                print('Steady state detected!')
                
                # need to make sure t_eval matches soln for output
                t_eval = t_eval[:len(soln)]
                
                break
        ############# write restart file ################## 
        if (i%checkpnt_freq==0):
            # write a restart file,
            # format is current t, followed by X arr
            np.savetxt('restrt_%08d.ascii'%(i), \
                       np.concatenate((np.array([t_arr[i]]),X_new)))
        
            
        # move the new values into X
        X = np.copy(X_new)

    # convert soln lists to arrays
    soln = np.array(soln)
    vel_U = np.array(vel_U)
    vel_W = np.array(vel_W)
    
    # indicies need swapping to match soln from ivp_solve
    soln = np.transpose(soln)
    vel_U = np.transpose(vel_U)
    vel_W = np.transpose(vel_W)
    
elif(method=='RK23' or method=='RK45' or method=='DOP853'): # use the scipy solve_ivp

    print('****************************')
    print('using ivp with method=%s'%(method))
    start = time.time()
    if (output_freq==1):
        # allow ivp_solve to output its own full soln
        soln_full = solve_ivp(lh.X_RHS, (t0,tf), X, args=(nnx, x, h), method=method, first_step=1e-6)
    else:
        # use subsampled t_arr points for evaluation
        soln_full = solve_ivp(lh.X_RHS, (t0,tf), X, args=(nnx, x, h), method=method, t_eval=t_eval)
    
    # produce separate soln and t_arrs to match output of Euler mode
    soln = soln_full.y
    t_arr = soln_full.t
    t_eval = t_arr

    print('nb of time steps=',len(t_arr))
    print('average time step=', tf/len(t_arr))

    delta_t=1
    vel_U=np.array(lh.U(soln[4*nnx:,:]))
    vel_W=np.array(lh.W(soln[4*nnx:,:]))
else:
    print('selected method is not a supported choice')
    print('Options are Euler, RK23, RK45, DOP853')
if (method=='Euler'):
    print('***********************istep=',i)
print('time elapsed=',  time.time()-start)

###########################################################################################

##### save results to vtu, ascii format

export_to_vtu2(len(t_eval), x, soln, vel_U, vel_W, lh.ADZ_bot, lh.ADZ_top, lh.x_scale, delta_t)

# Take time series at fixed depth
depths = [int(nnx/4), int(nnx/2), int(3*nnx/4),  nnx-1]
for i in depths:
    export_to_ascii('x', i, t_eval, soln[i,:], soln[nnx+i,:], soln[2*nnx+i,:],\
                    soln[3*nnx+i,:], soln[4*nnx+i,:], lh.U(soln[4*nnx+i,:]), lh.W(soln[4*nnx+i,:]))

# then take depth profiles at fixed time
times = [int(len(t_eval)/4), int(len(t_eval)/2),\
         int(3*len(t_eval)/4),  len(t_eval)-1]
for i in times:
    export_to_ascii('t', i, x, soln[0:nnx,i], soln[nnx:2*nnx,i], soln[2*nnx:3*nnx,i],\
                    soln[3*nnx:4*nnx,i], soln[4*nnx:5*nnx,i], lh.U(soln[4*nnx:5*nnx,i]), lh.W(soln[4*nnx:5*nnx,i]))







