import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed


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
    Lambda = conf['k0'] / (conf['D'] * f * conf['v']) ** 0.5  # dimensionless reversibility parameter (Eqn 6.4.4, pg. 236-239)
    # chem reversibility warning
    #if km > 0.1:
    #    print('k_c*t_k/l equals {} which exceeds the upper limit of 0.1 (see B&F, pg 797)'.format(km))

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
        JO[i1 + 1] = (kf[i1 + 1] * O[i1 + 1, 2] - kb[i1 + 1] * R[i1 + 1, 2]) / (1 + Dx / conf['D'] * (kf[i1 + 1] + kb[i1 + 1]))  #
        # Update surface concentrations
        O[i1 + 1, 1] = O[i1 + 1, 2] - JO[i1 + 1] * (Dx / conf['D'])  #
        R[i1 + 1, 1] = R[i1 + 1, 2] + JO[i1 + 1] * (Dx / conf['D']) - km * R[i1 + 1, 1]  #

    # Calculate current density, Z, from flux of O
    Z = -conf['n'] * F * JO / 10.  # A/m^2 -> mA/cm^2, current density
    return eta, Z, t


steps = 10
v = np.linspace(0.01, 1, steps)  # we're implicitly treating this as input
etai = 0.5
etaf = -0.5
k0 = np.logspace(-5, 1, steps)
kc = np.logspace(-5, 1, steps)
C = np.logspace(-8, 1, steps)
D = np.logspace(-8, 2, steps)

vgrid, k0grid, kcgrid, cgrid, dgrid = np.meshgrid(v, k0, kc, C, D)
vgrid, k0grid, kcgrid, cgrid, dgrid = vgrid.flatten(), k0grid.flatten(), kcgrid.flatten(), cgrid.flatten(), dgrid.flatten()
el, zl, tl = [], [], []


def run(v, k0, kc, c, d):
    conf = {'C': c,  # mol/cm^3, initial concentration of O. Default = 1.0
            'D': d,  # cm^2/s, O & R diffusion coefficient. Default = 1E-5
            'etai': +0.5,  # V, initial overpotential (relative to redox potential). Default = +0.2
            'etaf': -0.5,  # V, final overpotential (relative to redox potential). Default = -0.2
            'v': v,  # V/s, sweep rate. Default = 1E-3
            'n': 1.0,  # number of electrons transfered. Default = 1
            'alpha': 0.5,  # dimensionless charge-transfer coefficient. Default = 0.5
            'k0': k0,  # cm/s, electrochemical rate constant. Default = 1E-2
            'kc': kc,  # 1/s, chemical rate constant. Default = 1E-3
            'T': 298.15,  # K, temperature. Default = 298.15
            }
    e, z, t = sim_cv(conf)
    return [e, z, t]


#res = Parallel(n_jobs=-1)(delayed(run)(v, k0, kc, c, d) for v, k0, kc, c, d in tqdm(zip(vgrid, k0grid, kcgrid, cgrid, dgrid)))
