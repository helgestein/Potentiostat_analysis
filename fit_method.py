from sklearn.preprocessing import QuantileTransformer
import numpy as np
from sklearn.decomposition import PCA
import pickle
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score
from simulate_cyclic_voltammetry import sim_cv, run

data = pickle.load(open('C:/Users/Fuzhi/OneDrive/Share_folder/data/data.pck', 'rb'))

# train data
pca = PCA(n_components=6)
X_pca_train = pca.fit_transform(data['current'][data['train_ix']])
# var = pca.explained_variance_ratio_
sum(pca.explained_variance_ratio_[0:6])

scan_rate_train = np.reshape(data['v'][data['train_ix']], (-1, 1))

concentration_train = np.reshape(data['c'][data['train_ix']], (-1, 1))

train_x = np.append(np.append(X_pca_train, scan_rate_train, axis=1), concentration_train, axis=1)

# validation data

X_pca_val = pca.transform(data['current'][data['val']])

scan_rate_val = np.reshape(data['v'][data['val']], (-1, 1))

concentration_val = np.reshape(data['c'][data['val']], (-1, 1))

val_x = np.append(np.append(X_pca_val, scan_rate_val, axis=1), concentration_val, axis=1)

# test data

X_pca_test = pca.transform(data['current'][data['test_ix']])

scan_rate_test = np.reshape(data['v'][data['test_ix']], (-1, 1))

concentration_test = np.reshape(data['c'][data['test_ix']], (-1, 1))

test_x = np.append(np.append(X_pca_test, scan_rate_test, axis=1), concentration_test, axis=1)

# prediction

scaling = QuantileTransformer(n_quantiles=10, random_state=0)

k0_train = data['k0'][data['train_ix']]
kc_train = data['kc'][data['train_ix']]
D_train = data['d'][data['train_ix']]

y = np.array([k0_train, kc_train, D_train])
train_y = scaling.fit_transform(y.T)

k0_test = data['k0'][data['test_ix']]
kc_test = data['kc'][data['test_ix']]
D_test = data['d'][data['test_ix']]

yt = np.array([k0_test, kc_test, D_test])
test_y = scaling.transform(yt.T)

# validation
k0_val = data['k0'][data['val']]
kc_val = data['kc'][data['val']]
D_val = data['d'][data['val']]

yv = np.array([k0_test, kc_test, D_test])
val_y = scaling.transform(yv.T)

# multi random forest regression model
MultiRFRegression = MultiOutputRegressor(RandomForestRegressor(n_estimators=50, max_depth=45, min_samples_split=3, random_state=3))
MultiRFRegression.fit(train_x, train_y)
# MultiRFRegression.feature_importances_
y_predict = MultiRFRegression.predict(test_x)

for i in range(3):
    print(r2_score(test_y[:, i], y_predict[:, i]))
    # print(mse(test_y[:, i], y_predict[:, i]))
    plt.figure(figsize=(5,5))
    plt.hist2d(test_y[:,i], y_predict[:,i], bins= 10)
    #plt.hist2d(train_x[:,i], train_y[:,i], bins= 10)
    #plt.hist2d(test_y[:, 0], test_y[:, 1], edgecolor='k', label="Data")
    #plt.hist2d(y_predict[:, 0], y_predict[:, 1], edgecolor='k', label="Multi RF score=%.2f" % r2_score(test_y, y_predict))
    plt.xlim([0, 1.5])
    plt.ylim([0, 1.5])
    plt.axis('equal')
    plt.show()

# getting the index and investigating regarding the data'value
test_indices = np.where((test_y[:, 1] > 0.79) & (test_y[:, 1] < 0.91))[0]
global_test_indices = np.random.choice(test_indices, 100, replace=False)

# evaluating the data correspond to their indices

# kc_test_new = test_y[:,1][test_indices]
kc_test_new = data['kc'][global_test_indices]
# k0_test_new = test_y[:,0][test_indices]
k0_test_new = data['k0'][global_test_indices]
# d_test_new = test_y[:,2][test_indices]
d_test_new = data['d'][global_test_indices]


current_test = []
potential_test = []
for i in range(100):
    current_test.append(data['current'][global_test_indices[i]])
    potential_test.append(data['potential'][global_test_indices[i]])


# taking the new simulation for the test data
e1, z1, t1 = [], [], []
for i in range(100):
    conf = {'C': 1.0,  # mol/cm^3, initial concentration of O. Default = 1.0
            'D': d_test_new[i],  # cm^2/s, O & R diffusion coefficient. Default = 1E-5
            'etai': +0.5,  # V, initial overpotential (relative to redox potential). Default = +0.2
            'etaf': -0.5,  # V, final overpotential (relative to redox potential). Default = -0.2
            'v': 1E-3,  # V/s, sweep rate. Default = 1E-3
            'n': 1.0,  # number of electrons transfered. Default = 1
            'alpha': 0.5,  # dimensionless charge-transfer coefficient. Default = 0.5
            'k0': k0_test_new[i],  # cm/s, electrochemical rate constant. Default = 1E-2
            'kc': kc_test_new[i],  # 1/s, chemical rate constant. Default = 1E-3
            'T': 298.15,  # K, temperature. Default = 298.15
            }
    e, z, t = sim_cv(conf)
    e1.append(e)
    z1.append(z)
    t1.append(t)

# getting the index and investigating regarding the prediction data'value

indices_pred = np.where((y_predict[:, 1] > 0.79) & (y_predict[:, 1] < 0.91))[0]
global_pred_indices = np.random.choice(indices_pred, size=100, replace=False)

# evaluating the data correspond to their indices
kc_predict = y_predict[:, 1][global_pred_indices]
k0_predict = y_predict[:, 0][global_pred_indices]
d_predict = y_predict[:, 2][global_pred_indices]

'''
current_pred = []
potential_pred = []
for i in range(100):
    current_pred.append(data['current'][global_pred_indices[i]])
    potential_pred.append(data['potential'][global_pred_indices[i]])

fig, ax = plt.subplots(10, 10)
ax = ax.flatten()
for i in range(100):
    ax[i].plot(potential_pred[i], current_pred[i]/(np.max(np.abs(current_pred[i]))))
    ax[i].plot(potential_test[i], current_test[i]/np.max(np.abs(current_test[i])))
    ax[i].axis('off')
plt.show()
'''

# taking the new simulation for the prediction data

pred_simulation = []
for i in range(len(kc_predict)):
    pred_simulation.append(run(1E-3, k0_predict[i], kc_predict[i], 1.0, d_predict[i]))

fig, ax= plt.subplots(10, 10)
ax = ax.flatten()
for i in range(100):
    np.nan_to_num(pred_simulation[i][1], copy=False)
    #np.seterr(divide='ignore', invalid='ignore')
    ax[i].plot(e1[i], z1[i] / (np.max(np.abs(z1[i]))))
    ax[i].plot(pred_simulation[i][0], pred_simulation[i][1]/np.max(np.abs(pred_simulation[i][1])))
    ax[i].plot(potential_test[i], current_test[i]/np.max(np.abs(current_test[i])))
    ax[i].axis('off')
plt.show()
