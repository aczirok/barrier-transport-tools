import numpy as np

def sampling(I,x0=(0,0),kon=0.003, koff=0.103, q=0.127):
    """Model for sampling cross-contamination 
    Args:
        I: analyte input
        x0: initial state of hidden variables 
        kon, koff, q: parameters
    Returns:
        "output": analyte amounts in the sample sequence
        "hidden": analyte [in droplets, adsorbed]
    
    """
    samples=[]         #samples generated
    hidden =[]         #hidden state variable
    x=x0
    for i in I:
        T = i + x[0] + x[1]*koff
        S = T * (1-q)*(1-kon)
        y0= T * q * (1-kon)
        y1= x[1]*(1-koff) + T*kon
        x = (y0,y1)
        samples.append(S)
        hidden.append(x)
    return {
        "output": samples,
        "hidden": hidden
    }

import numpy as np
from scipy.integrate import solve_ivp

A = 0.6*100                        #[mm^2] membrane area
h_A = 250/A                        #[mm], height of apical volume
h_B = h_A * 1.2/.25                #[mm], effective height of basolateral volume
t_samples = np.linspace(0,60*6.5,14) #6 hours, 30 min sampling period

def membrane(c_A0=10, P0=.1):
    """returns analyte amounts in the sample sequence.
    Args:
        c_A0: intial _concentration_ in apical volume
        P0  : membrane permeability [mm/min]""" 
    def ode(t,y):
        c_A, c_B = y
        J    = P0*(c_A - c_B)
        dc_A = -J/h_A
        dc_B = J/h_B
        return [dc_A, dc_B]
    
    samples=[]         #samples generated
    #initial condition
    t1 = 0
    y0 = [c_A0, 0]
    for t2 in t_samples[1:]:
        tspan = (0, t2-t1)
        sol = solve_ivp(ode, tspan, y0)
        c_A, c_B = sol.y[:, -1]
        samples.append(c_B)
        t1=t2
        y0 = [c_A, 0]
    return {
        "output": np.array(samples) *1.2    #[nmol]
    }


import numpy as np
from scipy.integrate import solve_ivp

A = 0.6*100                        #[mm^2] membrane area
h_A = 250/A                        #[mm], height of apical volume
h_B = h_A * 1.2/.25                #[mm], effective height of basolateral volume
t_samples = np.linspace(0,60*6.5,14) #6 hours, 30 min sampling period

def cell(c_A0=20, P0=.05, P1=0.1, h=0.01, m=0):

    def ode(t,y):
        c_A, c_B, c_C = y
        
        J_A  = P1 * (c_A - c_C)
        J_B  = P1 * (c_C - c_B )/(1 + P1/P0) 
        
        dc_A = -J_A/h_A
        dc_B =  J_B/h_B
        dc_C = (J_A - J_B - h*m*c_C )/h
        
        return([dc_A, dc_B, dc_C])
    
    samples=[]         #samples generated
    
    #initial condition
    t1=0
    y0 = [c_A0, 0, 0]
    
    for t2 in t_samples[1:]: 
        tspan = (0, t2-t1)
        
        sol = solve_ivp(ode, tspan, y0)
        
        c_A, c_B, c_C = sol.y[:, -1]
        samples.append(c_B*1.2)        
        t1=t2
        y0 = [c_A, 0, c_C]
    samples.append(c_A*.25)
        
    return {
        "output": np.array(samples) *1.2    #[nmol]
    }

##########general fitting process##############
import numpy as np
from scipy.optimize import minimize
def fitlog(logfile, *args):
    if logfile:
        with open(logfile, "a") as f:
            print(*args, file=f)

def fit(process, target, score, pvar, 
        constrain=lambda p: False,
        localize=lambda p: 0,
        pfix=None, logfile=False, save=False, method='BFGS'):
    """ fits a process to a target.
    Args:
        process(pvar,pfix): function to be fitted. 
            returns a dictionary, with a mandatory key "output" 
        target: target object 
        score: function to compare the output of process with the target object
        pvar: parameters to be optimized
        pfix: parameters to be kept fixed at a value distinct from the default
        constrain: penalty function to evaluate pvar
        localize:  penalty function to evaluate pvar
        logfile: file to log the progress of the fitting process
        save: a file to save the result of the fitting process
        
    Returns:
        "params": the optimized parameters, as a dictionary like pvar
        "output": the output object, comparable with target
        "error": value of the minimized objective function
    """
    fitlog(logfile,"------")
    
    if isinstance(pvar, dict):
        optimize = list(pvar.keys())
        p0       = list(pvar.values())
        from_dict = True
    else:
        p0       = list(pvar)
        from_dict = False
    
    param_dict = {**(pfix or {})}   #param_dict and pfix should not be a shared ref
    
    def objective(p):
        """function called by scipy.optimize.minimize
        Args: a flat list of parameters to be changed
        as a nested function it knows about the other params provided to fit
        """
        if constrain(p): return 1e6
        penalty = localize(p)
        
        if from_dict:
            param_dict.update(dict(zip(optimize,p)))            
            P = process(**param_dict)
        else:            
            P = process(p, **param_dict)            
        S = score( P["output"], target )
        fitlog(logfile, p, S, penalty)
        return S + penalty
    
    res = minimize(objective, p0, method=method)
    fitlog(logfile, res)
    if from_dict:
        p1 = dict(zip(optimize,res.x))
        param_dict.update(p1)
        P = process(**param_dict)
    else:
        p1 = res.x
        P = process(p1, **param_dict)
    P.update({
        "params": p1,
        "error" : res.fun
    })
    return P

###############
