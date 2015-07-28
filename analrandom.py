# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 15:20:47 2015

@author: yaron
"""
#%%
import math
from functools import partial

import numpy as np
import matplotlib.pyplot as plt
from lmfit import minimize, Parameters, fit_report
from numpy import linalg as LA
#%%


def model1(dataOligomer, pars):
    parvals = pars.valuesdict()
    alpha = parvals['alpha']
    endRaise = parvals['delta']
    beta = parvals['beta']
    npow = parvals['npow']
    angs = dataOligomer['angles']
    oligLength = angs.size + 1
    coupling = beta * np.abs(np.power(np.cos(np.deg2rad(angs)), npow))
    hamil = np.diag( np.full(oligLength   , alpha), 0 ) + \
        np.diag( coupling                     , -1)  + \
        np.diag(coupling, 1)
    hamil[0, 0] += endRaise
    hamil[-1, -1] += endRaise
    evals = LA.eig(hamil)
    return np.min(evals[0])


def model3(dataOligomer, pars):
    parvals = pars.valuesdict()
    alpha = parvals['alpha']
    endRaise = parvals['delta']
    const = parvals['const']
    beta = parvals['beta']
    beta2 = parvals['beta2']
    angs = dataOligomer['angles']
    oligLength = angs.size + 1
    coupling = const \
        + beta * np.abs(np.cos(np.deg2rad(angs))) \
        + beta2 * np.power(np.cos(np.deg2rad(angs)), 2)
    hamil = np.diag( np.full(oligLength   , alpha), 0 ) + \
        np.diag( coupling                     , -1)  + \
        np.diag(coupling, 1)
    hamil[0, 0] += endRaise
    hamil[-1, -1] += endRaise
    evals = LA.eig(hamil)
    return np.min(evals[0])


def angle_to_coupling(theta, betas):
    if theta > 180.1:
        theta = abs(theta - 360)
    itheta = int(math.floor(round(theta / 10)))
    # print itheta
    # print betas
    return betas[itheta]


def model2(dataOligomer, pars):
    parvals = pars.valuesdict()
    alpha = parvals['alpha']
    endRaise = parvals['delta']
    betas = []
    for i in xrange(0, 181, 10):
        betas.append(parvals['beta%i' % i])
    angs = dataOligomer['angles']
    oligLength = angs.size + 1
    coupling = []
    for ang in angs:
        coupling.append(angle_to_coupling(ang, betas))
    hamil = np.diag( np.full(oligLength   , alpha), 0 ) + \
        np.diag( coupling                     , -1)  + \
        np.diag(coupling, 1)
    hamil[0, 0] += endRaise
    hamil[-1, -1] += endRaise
    evals = LA.eig(hamil)
    return np.min(evals[0])


def residual(model, pars, x, data, eps_data=None):
    res = []
    for d in data:
        exc = d['exc']
        excPred = model(d, pars)
        res.append(excPred - exc)
    return np.asarray(res)


def write_statistics(name, outf, pars, fit_result):
    outf.write(name)
    outf.write(fit_report(pars))
    outf.write('\n')
    outf.write('Norm residuals is ' + repr(LA.norm(fit_result.residual)))
    outf.write('\n')
    outf.write('Mean absolute error ' +
               repr(np.mean(np.abs(fit_result.residual))))
    outf.write('\n')
    outf.write('-------------\n')


def plot_something(model, pars, exc, title="", filename="", interval=1):
    plt.figure(1)
    plt.clf()

    fmts = ['r', 'b', 'g']
    for i in xrange(1, 4):
        plt.plot(exc[:, 0], exc[:, i], fmts[i - 1] +
                 'o-', label='N=%d TDDFT' % (i + 1))

    plt.ylabel('Excitation energy')
    plt.xlabel('Angle')
    plt.title(title)

    for nolig in xrange(2, 5):
        x = []
        y = []
        for ang in xrange(0, 361, interval):
            x.append(ang)
            d = {'angles': np.full(nolig - 1, ang)}
            y.append(model(d, pars))
        plt.plot(x, y, fmts[nolig - 2] + '--', label='N=%d model' % nolig)
    plt.legend()
    plt.savefig(filename)


    beta_res = [0, 0]
    for cos_pow in [1, 2]:
        # Set up paramters
        pars = Parameters()
        pars.add('alpha',    value=5.0, vary=True)
        pars.add('beta',     value=-1.0, vary=True)
        pars.add('delta',    value=0.0, vary=end_diff)
        pars.add('npow',     value=cos_pow, vary=False)

        residual1 = partial(residual, model1)
        fit_result = minimize(residual1, pars, args=([], fitData, []))
        if not end_diff:
            parvals = pars.valuesdict()
            beta_res[cos_pow - 1] = parvals['beta']

        name = 'Thiophene: cos power=%d, delta varied=%r\n' % (
            cos_pow, end_diff)
        write_statistics(name, outf, pars, fit_result)

        title = 'Coupling = cosine to power %d ' % cos_pow
        plot_something(model1, pars, exc, title=title, filename='thio_rg_cos%d%r.png' % (cos_pow, end_diff))
    return beta_res


def model3_fits(fitData, exc, outf, end_diff=False):
    # Set up paramters
    pars = Parameters()
    pars.add('alpha',    value=5.0, vary=True)
    pars.add('const',    value=0.0, vary=True)
    pars.add('beta',     value=0.0, vary=False)
    pars.add('beta2',    value=-1.0, vary=True)
    pars.add('delta',    value=0.0, vary=end_diff)

    residual3 = partial(residual, model3)
    fit_result = minimize(residual3, pars, args=([], fitData, []))
    parvals = pars.valuesdict()
    beta_mod3 = (parvals['const'], parvals['beta'], parvals['beta2'])

    name = 'Thiophene model3: delta varied=%r\n' % end_diff
    write_statistics(name, outf, pars, fit_result)

    title = "Coupling = const + beta2 cos^2(theta) \n"
    plot_something(model3, pars, exc, title=title, filename='thio_mod3%r.png' % (end_diff))
    return beta_mod3


def discrete_fits(fitData, exc, outf, end_diff=False):
    # Set up paramters
    pars = Parameters()
    pars.add('alpha',    value=5.0, vary=True)
    for i in xrange(0, 181, 10):
        pars.add('beta%i' % i, value=-0.8 *
                 np.abs(math.cos(i * 3.14 / 180.0)), vary=True)
    pars.add('delta',    value=0.0, vary=end_diff)

    residual2 = partial(residual, model2)
    fit_result = minimize(residual2, pars, args=([], fitData, []))

    name = 'Thiophene discrete: delta varied=%r\n' % end_diff
    write_statistics(name, outf, pars, fit_result)

    title = "Treating beta at each angle as a separate parameter \n"
    plot_something(model2, pars, exc, title=title, filename='thio_discrete%r.png' % (end_diff), interval=10)
    return pars




if __name__ == "__main__":
    """
    ang = exc[:,0]
    e2  = exc[:,1]
    e3  = exc[:,2]
    e4  = exc[:,3]
    """
    exc = np.loadtxt(open('thio_uniform.csv', 'r'), delimiter=",")
    ran3 = np.loadtxt(open('thio_random_3.csv', 'r'), delimiter=",", skiprows=1)
    ran4 = np.loadtxt(open('thio_random_4.csv', 'r'), delimiter=",", skiprows=1)

    # angles is list of angles for the oligomer
    fitData = []
    for angLength in xrange(1, 4):
        for angIndex in xrange(exc[:, 0].size):
            if (np.abs(np.cos(np.deg2rad(exc[angIndex, 0]))) > 0.05):
                fitData.append(
                    {'angles': np.full(angLength, exc[angIndex, 0]),
                     'exc': exc[angIndex, angLength]}
                )
    for row in ran3:
        fitData.append(
            {'angles': row[:2],
             'exc': row[2]}
        )
    for row in ran4:
        fitData.append(
            {'angles': row[:3],
             'exc': row[3]}
        )

    do_cos_fits = True
    do_discrete_fits = True
    do_model3 = True
    with open('randomFits.txt', 'w') as outf:
        if do_cos_fits:
            beta_res = cos_fits(fitData, exc, outf)
        if do_model3:
            beta_mod3 = model3_fits(fitData, exc, outf)
        if do_discrete_fits:
            pars = discrete_fits(fitData, exc, outf)

        parvals = pars.valuesdict()
        x_theta = []
        y_beta = []
        y_cos = []
        y_cos2 = []
        for i in xrange(0, 181, 10):
            x_theta.append(i)
            y_beta.append(parvals['beta%i' % i])
        y_cos = beta_res[0] * np.abs(np.cos(np.deg2rad(x_theta)))
        y_cos2 = beta_res[0] * np.power(np.cos(np.deg2rad(x_theta)), 2)
        y_mod3 = beta_mod3[0] \
            + beta_mod3[1] * np.abs(np.cos(np.deg2rad(x_theta))) \
            + beta_mod3[2] * np.power(np.cos(np.deg2rad(x_theta)), 2)
        plt.figure(2)
        plt.title('Comparision of beta(theta) across models\n')
        plt.plot(x_theta, y_beta, 'ro')
        plt.plot(x_theta, y_cos, 'b-')
        plt.plot(x_theta, y_cos2, 'g-')
        plt.plot(x_theta, y_mod3, 'k-')
        plt.show()
