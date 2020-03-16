from copy import copy
import matplotlib.pyplot as plt
import numpy as np
from lmfit import Minimizer, Parameters, report_fit, Model
from scipy.interpolate import interp1d



conf = {'C': 1.0,  # mol/cm^3, initial concentration of O. Default = 1.0
        'D': 1E-5,  # cm^2/s, O & R diffusion coefficient. Default = 1E-5
        'etai': +0.2,  # V, initial overpotential (relative to redox potential). Default = +0.2
        'etaf': -0.2,  # V, final overpotential (relative to redox potential). Default = -0.2
        'v': 0.02,  # V/s, sweep rate. Default = 1E-3
        'n': 1.0,  # number of electrons transfered. Default = 1
        'alpha': 0.5,  # dimensionless charge-transfer coefficient. Default = 0.5
        'k0': 1E-2,  # cm/s, electrochemical rate constant. Default = 1E-2
        'kc': 1E-3,  # 1/s, chemical rate constant. Default = 1E-3
        'T': 298.15,  # K, temperature. Default = 298.15
        'offset': 0.01
        }


def sim_cv(conf, L=1000, DM=0.45):
    F = 96485  # C/mol, Faraday's constant
    R = 8.3145  # J/mol-K, ideal gas constant

    # L number of iterations per t_k (pg 790). Default = 500
    # DM model diffusion coefficient (pg 788). Default = 0.45

    # normalized faraday constant
    f = F / (R * conf['T'])  # 1/V, normalized Faraday's constant at room temperature

    # derived constants
    tk = 2 * (conf['etai'] - conf['etaf']) / conf['v']  # s, characteristic exp. time (pg 790).
    # In this case, total time of fwd and rev scans
    Dt = tk / L  # s, delta time (Eqn B.1.10, pg 790)
    Dx = np.sqrt(conf['D'] * Dt / DM)  # cm, delta x (Eqn B.1.13, pg 791)
    j = np.ceil(4.2 * L ** 0.5) + 5  # number of boxes (pg 792-793). If L~200, j=65

    # reversibility parameters
    ktk = conf['kc'] * tk  # dimensionless kinetic parameter (Eqn B.3.7, pg 797)
    km = ktk / L  # normalized dimensionless kinetic parameter (see bottom of pg 797)
    Lambda = conf['k0'] / (
            conf['D'] * f * conf['v']) ** 0.5  # dimensionless reversibility parameter (Eqn 6.4.4, pg. 236-239)

    # chem reversibility warning
    if km > 0.1:
        print('k_c*t_k/l equals {} which exceeds the upper limit of 0.1 (see B&F, pg 797)'.format(km))

    # pre initialization
    k = np.array([i for i in range(L)])  # time index vector
    t = Dt * k  # time vector
    eta1 = conf['etai'] - conf['v'] * t  # overpotential vector, negative scan
    eta2 = conf['etaf'] + conf['v'] * t  # overpotential vector, positive scan
    eta = np.append(eta1[eta1 > conf['etaf']], eta2[eta2 <= conf['etai']])  # overpotential scan, both directions

    O = conf['C'] * np.ones([L + 1, int(j)])  # mol/cm^3, concentration of O
    R = np.zeros([L + 1, int(j)])  # mol/cm^3, concentration of R
    JO = np.zeros(L + 1)  # mol/cm^2-s, flux of O at the surface

    Enorm = eta * f  # normalized overpotential
    kf = conf['k0'] * np.exp(-conf['alpha'] * conf['n'] * Enorm)  # cm/s, fwd rate constant (pg 799)
    kb = conf['k0'] * np.exp((1 - conf['alpha']) * conf['n'] * Enorm)  # cm/s, rev rate constant (pg 799)

    # START SIMULATION %%
    # i1 = time index. i2 = distance index
    for i1 in range(L):  # !1:l
        # Update bulk concentrations of O and R
        for i2 in range(int(j) - 1):  # !i2  2:j-1
            O[i1 + 1, i2] = O[i1, i2] + DM * (O[i1, i2 + 1] + O[i1, i2 - 1] - 2 * O[i1, i2])  #
            R[i1 + 1, i2] = R[i1, i2] + DM * (R[i1, i2 + 1] + R[i1, i2 - 1] - 2 * R[i1, i2]) - km * R[i1, i2]
        # Update flux
        JO[i1 + 1] = (kf[i1 + 1] * O[i1 + 1, 2] - kb[i1 + 1] * R[i1 + 1, 2]) / (
                1 + Dx / conf['D'] * (kf[i1 + 1] + kb[i1 + 1]))  #
        # Update surface concentrations
        O[i1 + 1, 1] = O[i1 + 1, 2] - JO[i1 + 1] * (Dx / conf['D'])  #
        R[i1 + 1, 1] = R[i1 + 1, 2] + JO[i1 + 1] * (Dx / conf['D']) - km * R[i1 + 1, 1]  #

    # Calculate current density, Z, from flux of O
    Z = -conf['n'] * F * JO / 10.  # A/m^2 -> mA/cm^2, current density
    return eta, Z, t


def sim_multi_cv(conf, L=1000, DM=0.45, p=10, phi=2.5):
    triangle = lambda p, t: 2 / np.pi * np.arcsin(np.sin(2 * np.pi / p * t))
    from scipy import signal
    F = 96485  # C/mol, Faraday's constant
    R = 8.3145  # J/mol-K, ideal gas constant

    # L number of iterations per t_k (pg 790). Default = 500
    # DM model diffusion coefficient (pg 788). Default = 0.45

    # normalized faraday constant
    f = F / (R * conf['T'])  # 1/V, normalized Faraday's constant at room temperature

    # derived constants
    tk = 2 * (conf['etai'] - conf['etaf']) / conf['v']  # s, characteristic exp. time (pg 790).
    # In this case, total time of fwd and rev scans
    Dt = tk / L  # s, delta time (Eqn B.1.10, pg 790)
    Dx = np.sqrt(conf['D'] * Dt / DM)  # cm, delta x (Eqn B.1.13, pg 791)
    j = np.ceil(4.2 * L ** 0.5) + 5  # number of boxes (pg 792-793). If L~200, j=65

    # reversibility parameters
    ktk = conf['kc'] * tk  # dimensionless kinetic parameter (Eqn B.3.7, pg 797)
    km = ktk / L  # normalized dimensionless kinetic parameter (see bottom of pg 797)
    Lambda = conf['k0'] / (
            conf['D'] * f * conf['v']) ** 0.5  # dimensionless reversibility parameter (Eqn 6.4.4, pg. 236-239)

    # chem reversibility warning
    if km > 0.1:
        print('k_c*t_k/l equals {} which exceeds the upper limit of 0.1 (see B&F, pg 797)'.format(km))

    # pre initialization
    k = np.array([i for i in range(L + 1)])  # time index vector
    t = Dt * k  # time vector
    # eta1 = conf['etai'] - conf['v']*t# overpotential vector, negative scan
    # eta2 = conf['etaf'] + conf['v']*t# overpotential vector, positive scan
    eta = conf['etai'] * triangle(p, t - phi) + conf['offset']
    # eta = np.append(eta,eta[-1])
    O = conf['C'] * np.ones([L + 1, int(j)])  # mol/cm^3, concentration of O
    R = np.zeros([L + 1, int(j)])  # mol/cm^3, concentration of R
    JO = np.zeros(L + 1)  # mol/cm^2-s, flux of O at the surface

    Enorm = eta * f  # normalized overpotential
    kf = conf['k0'] * np.exp(-conf['alpha'] * conf['n'] * Enorm)  # cm/s, fwd rate constant (pg 799)
    kb = conf['k0'] * np.exp((1 - conf['alpha']) * conf['n'] * Enorm)  # cm/s, rev rate constant (pg 799)

    # START SIMULATION %%
    # i1 = time index. i2 = distance index
    for i1 in range(L):  # !1:l
        # Update bulk concentrations of O and R
        for i2 in range(int(j) - 1):  # !i2  2:j-1
            O[i1 + 1, i2] = O[i1, i2] + DM * (O[i1, i2 + 1] + O[i1, i2 - 1] - 2 * O[i1, i2])  #
            R[i1 + 1, i2] = R[i1, i2] + DM * (R[i1, i2 + 1] + R[i1, i2 - 1] - 2 * R[i1, i2]) - km * R[i1, i2]
        # Update flux
        JO[i1 + 1] = (kf[i1 + 1] * O[i1 + 1, 2] - kb[i1 + 1] * R[i1 + 1, 2]) / (
                1 + Dx / conf['D'] * (kf[i1 + 1] + kb[i1 + 1]))  #
        # Update surface concentrations
        O[i1 + 1, 1] = O[i1 + 1, 2] - JO[i1 + 1] * (Dx / conf['D'])  #
        R[i1 + 1, 1] = R[i1 + 1, 2] + JO[i1 + 1] * (Dx / conf['D']) - km * R[i1 + 1, 1]  #

    # Calculate current density, Z, from flux of O
    Z = -conf['n'] * F * JO / 10.  # A/m^2 -> mA/cm^2, current density
    return eta, Z, t


def fitfunction(t_, D=2, k0=0.01, kc=0.01):
    conf_residual = copy(conf)
    conf_residual['D'] = D
    conf_residual['k0'] = k0
    conf_residual['kc'] = kc
    V, I, t = sim_multi_cv(conf_residual, p=2, L=1000, phi=-2.5)  # periodicy (different scan speed)
    # use interpolation!
    f = interp1d(t, I, kind='cubic')

    return f(t_)



for i in range(10):
    # demonstrate multiple cv curves

    conf_ = copy(conf)
    conf_['v'] = 0.05
    conf_['offset'] = 0

    eta_tl, Z_tl, t = sim_multi_cv(conf_, p=2, L=1000, phi=-2.5)
    plt.plot(eta_tl, Z_tl, label=r'Multiple sweeps')
    plt.xlabel('Potential [V vs. Redox Potential]')
    plt.ylabel('Current Density [mA/cm^2]')
    plt.legend()
    plt.show()
    print(eta_tl)
    print(Z_tl)
    print(t)


    model = Model(fitfunction, independent_vars=['t_'])
    model.set_param_hint('D', value=0.006, min=1E-11, max=10)
    model.set_param_hint('k0', value=0.098, min=1E-10, max=10E+13)
    model.set_param_hint('kc', value=0.015, min=1E-11, max=10E+13)

    params = model.make_params(D=1, k0=0.1, kc=0.001)

    result = model.fit(Z_tl, t_= t, method='leastsq')
    print(result.values)
    print(result.fit_report())
    #print(result.params.pretty_print())
    conf['D'] = result.values.get('D')
    conf['k0'] = result.values.get('k0')
    conf['kc'] = result.values.get('kc')

'''


result.plot()
plt.figure()
#plt.plot()
#plt.plot(result.best_fit)
plt.show()


fit_method = ['leastsq', 'nelder', 'lbfgsb', 'powell', 'cg', 'newton', 'cobyla', 'bfgsb', 'tnc', '	trust-ncg', 'trust-exact', 'trust-krylov', 'trust-constr', 'dogleg', 'slsqp', 'differential_evolution', 'brute', 'basinhopping', 'ampgo', 'shgo', 'dual_annealing', 'emcee' ]
'''