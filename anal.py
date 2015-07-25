# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 15:20:47 2015

@author: yaron
"""
#%%
import numpy as np
import matplotlib.pyplot as plt
from lmfit import minimize, Parameters, fit_report
from numpy import linalg as LA
#%%

def model1(dataOligomer, pars):
    parvals = pars.valuesdict();
    alpha = parvals['alpha']
    endRaise =parvals['endRaise']
    beta  = parvals['beta']
    npow = parvals['npow']
    angs = dataOligomer['angles']
    oligLength = angs.size + 1
    coupling = beta *   np.power(np.cos(np.deg2rad(angs)), npow) 
    hamil = np.diag( np.full(oligLength   , alpha), 0 ) + \
            np.diag( coupling                     , -1)  + \
            np.diag( coupling                     , 1) 
    hamil[0,0]   += endRaise
    hamil[-1,-1] += endRaise
    evals = LA.eig(hamil)
    return np.min(evals[0])
  

def residual1(pars, x, data, eps_data=None):
    res = []
    for d in data:
        exc  = d['exc']
        excPred = model1(d, pars)
        res.append(excPred - exc)
    return np.asarray(res)
        
"""
             
ang = exc[:,0]
e2  = exc[:,1]
e3  = exc[:,2]
e4  = exc[:,3]
"""
with open('uniformFits.txt','w') as outf:

    for file_name in ["thio", "phenyl"]:
        exc = np.loadtxt(open(file_name+"Anal.csv","r"),delimiter=",")
        for cos_pow in [1 , 2]:
    
            
            plt.figure(1)
            plt.clf()
            fmts = ['r','b','g']
            
            for i in range(1,4):
                plt.plot(exc[:,0],exc[:,i], fmts[i-1]+'o-', label='N=%d TDDFT'%(i+1) )
            
            plt.ylabel('Excitation energy')
            plt.xlabel('Angle')
            plt.title(file_name + ' using cosine to power %d'%cos_pow)
            
            #%% Prepare data for fits
            # angles is list of angles for the oligomer
            uniformData = []
            for angLength in range(1,3):
                for angIndex in range( exc[:,0].size ):
                    if ( np.abs(np.cos(np.deg2rad(exc[angIndex,0]))) > 0.05):
                        uniformData.append(
                            {'angles' : np.full(angLength, exc[angIndex,0]), 
                             'exc'    : exc[angIndex, angLength]}
                             )
            
            
            # Set up paramters
            pars = Parameters()
            pars.add('alpha',    value= 5.0, vary=True)
            pars.add('endRaise',  value= 0.0, vary=False)
            pars.add('beta',     value=-1.0, vary=True)
            pars.add('npow',     value=cos_pow, vary=False)
            
            #b = residual1(pars, [], uniformData)
            
            fit_result = minimize(residual1, pars, args=([], uniformData, []))
            
            outf.write(file_name + ' cos^%d \n'%cos_pow)
            outf.write(fit_report(pars))
            outf.write('\n')
            outf.write('Norm residuals is ' + repr(LA.norm(fit_result.residual)))
            outf.write('\n')
            
            for nolig in range(2,5):
                x = [];
                y = [];
                for ang in range(0,361,1):
                    x.append(ang)
                    d = {'angles' : np.full(nolig-1, ang)}
                    y.append(model1(d,pars))
                plt.plot(x,y, fmts[nolig-2]+'--', label='N=%d model'%nolig)
            plt.legend()
            plt.savefig(file_name+'_cos^%d.eps'%cos_pow)
            
