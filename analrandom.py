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
    endRaise =parvals['delta']
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
  

def model2(dataOligomer, pars):
    parvals = pars.valuesdict();
    alpha = parvals['alpha']
    endRaise =parvals['delta']
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
with open('randomFits.txt','w') as outf:


    exc  = np.loadtxt(open('thio_uniform.csv','r') ,delimiter=",")
    ran3 = np.loadtxt(open('thio_random_3.csv','r'),delimiter=",", skiprows=1)
    ran4 = np.loadtxt(open('thio_random_4.csv','r'),delimiter=",", skiprows=1)
    
    for cos_pow in [1 , 2]:
        for end_diff in [True , False]:
        
            plt.figure(1)
            plt.clf()
            fmts = ['r','b','g']
            
            for i in range(1,4):
                plt.plot(exc[:,0],exc[:,i], fmts[i-1]+'o-', label='N=%d TDDFT'%(i+1) )
            
            plt.ylabel('Excitation energy')
            plt.xlabel('Angle')
            plt.title('fit random geoms using cosine to power %d end site raised: %r'%(cos_pow , end_diff))
            
            # angles is list of angles for the oligomer
            fitData = []
            for angLength in range(1,4):
                for angIndex in range( exc[:,0].size ):
                    if ( np.abs(np.cos(np.deg2rad(exc[angIndex,0]))) > 0.05):
                        fitData.append(
                            {'angles' : np.full(angLength, exc[angIndex,0]), 
                             'exc'    : exc[angIndex, angLength]}
                             )
                             
            for row in ran3:
                fitData.append(
                    {'angles' : row[:2] ,
                     'exc'    : row[2]}
                     )
            for row in ran4:
                fitData.append(
                    {'angles' : row[:3] ,
                     'exc'    : row[3]}
                     )
               
            # Set up paramters
            pars = Parameters()
            pars.add('alpha',    value= 5.0, vary=True)
            pars.add('beta',     value=-1.0, vary=True)
            pars.add('delta',    value= 0.0, vary=end_diff)
            pars.add('npow',     value=cos_pow, vary=False)
            
            #b = residual1(pars, [], uniformData)
            
            fit_result = minimize(residual1, pars, args=([], fitData, []))
            
            outf.write('Thiophene: cos power=%d, delta varied=%r\n'%(cos_pow,end_diff))
            outf.write(fit_report(pars))
            outf.write('\n')
            outf.write('Norm residuals is ' + repr(LA.norm(fit_result.residual)))
            outf.write('\n')
            outf.write('-------------\n')
            
            for nolig in range(2,5):
                x = [];
                y = [];
                for ang in range(0,361,1):
                    x.append(ang)
                    d = {'angles' : np.full(nolig-1, ang)}
                    y.append(model1(d,pars))
                plt.plot(x,y, fmts[nolig-2]+'--', label='N=%d model'%nolig)
            plt.legend()
            plt.savefig('thio_rg_cos%d%r.eps'%(cos_pow,end_diff))
            
