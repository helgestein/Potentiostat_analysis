import json
from simulate_cyclic_voltammetry import sim_multi_cv
import numpy as np
from lmfit import Minimizer, Parameters, report_fit, Model, Parameter
from pandas import Series

# a*(x-b)**2
# a*x**2

conf = {'C': 1.0,  # mol/cm^3, initial concentration of O. Default = 1.0
        'D': 1E-5,  # cm^2/s, O & R diffusion coefficient. Default = 1E-5
        'etai': +0.2,  # V, initial overpotential (relative to redox potential). Default = +0.2
        'etaf': -0.2,  # V, final overpotential (relative to redox potential). Default = -0.2
        'v': 0.02,  # V/s, sweep rate. Default = 1E-3
        'n': 1.0,  # number of electrons transfered. Default = 1
        'alpha': 0.5,  # dimensionless charge-transfer coefficient. Default = 0.5
        'k0': 1E-2,  # cm/s, electrochemical rate constant. Default = 1E-2
        'kc': 1E-3,  # 1/s, chemical rate constant. Default = 1E-3
        'T': 298.15  # K, temperature. Default = 298.15
        }

with open('cv_data.json') as f:
    data = json.load(f)
    original = np.array(data['current'])
print(original)


# define objective function: returns the array to be minimized
def simwrapper(params, original):
    """Model a decaying sine wave and subtract data."""
    d = params['D']
    k0 = params['k0']
    kc = params['kc']

    conf['D'] = d
    conf['ko'] = k0
    conf['kc'] = kc

    simdata = sim_multi_cv(conf)

    return simdata[1] - original


params = Parameters()
params.add('D', value=conf['D'])
params.add('k0', value=conf['k0'])
params.add('kc', value=conf['kc'])

minner = Minimizer(simwrapper, params, fcn_args=(original))
result = minner.minimize()
final = original + result.residual

report_fit(result)
try:
    import matplotlib.pyplot as plt
    plt.plot(data)
    plt.plot(final)
    plt.show()
except ImportError:
    pass

'''
model = Model(simwrapper(params))
result = model.fit(data['current'])
print(result.values)
result.plot

# do fit, here with the default leastsq algorithm
minner = Minimizer(fcn2min, params, fcn_args=(x, data))
result = minner.minimize()

# calculate final result
final = data + result.residual

# write error report
report_fit(result)

# try to plot results
try:
    import matplotlib.pyplot as plt
    plt.plot(x, data, 'k+')
    plt.plot(x, final, 'r')
    plt.show()
except ImportError:
    pass
    
'''



'''
import json

elec_info = dict()
key = ['potential', 'current', 'time']
elec_info['potential'] = eta_tl.tolist()
elec_info['current'] = Z_tl.tolist()
elec_info['time'] = t.tolist()
#elec_info.update(conf)

with open('cv_data.json', 'w') as json_file:
    json.dump(elec_info, json_file)

from lmfit import Minimizer, Parameters, report_fit, Model, Parameter

with open('cv_data.json') as f:
    data = json.load(f)
    original = np.array(data['current'])
print(original)

'''
